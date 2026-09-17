"""
Simple inference interface for tooth segmentation + landmark prediction.

Takes a single scan (.stl/.obj) or a folder that is scanned recursively, and
writes for every scan:
  - <stem>_seg.npy        cleaned per-vertex segmentation mask
  - <stem>_landmarks.ply  landmark points, one color per landmark class (only with --save-ply)
  - <stem>_segmentation_views.png  (only with --vis-seg)
  - <stem>_segmentation.ply       colored 3D segmentation mesh (only with --vis-seg-3d)
When the input is a folder, its directory structure is replicated inside the
output folder; a single scan is saved directly in the output folder.

At the end of the run, a single landmarks.json is also written at the root of
the output folder, aggregating every scan's raw landmark predictions (points
+ basePlane per tooth), flat and keyed by tooth_key exactly like
postprocess_predictions' own per-scan output (e.g. "118_lower_FDI_47").

Example:
python infer.py \
    --input /path/to/scans --output /path/to/predictions \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  application/app_weights/segmentator_best.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight application/app_weights/heatmap_landmarks.pth \
    --preprocessing preprocessing/3dteethland_preprocessing.yaml
"""
import os
import sys
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import json
import re
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from application.bond import ALL_LANDMARKS
from application.cache import TeethCache
from application.visualizers import json_to_ply
from application.pipeline import LandmarksPredictor

SCAN_EXTENSIONS = {".stl", ".obj"}
# Tooth-key parsing (application/utils.parse_tooth) expects a scan stem shaped
# like "<id>_<lower|upper>" or "STEM_<lower|upper>_<id>"; scans already in this
# form are used unchanged so tooth keys stay clean (e.g. "118_lower_FDI_47").
SCAN_NAME_RE = re.compile(r"^([^_]+_(lower|upper)|STEM_(lower|upper)_[^_]+)$")


def collect_scans(input_path: Path) -> list[Path]:
    """Return the single input scan, or all scans found recursively in a folder."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SCAN_EXTENSIONS:
            sys.exit(f"❌ Unsupported scan format: {input_path.suffix} (expected .stl/.obj)")
        return [input_path]
    if not input_path.is_dir():
        sys.exit(f"❌ Input path does not exist: {input_path}")
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SCAN_EXTENSIONS)


def output_dir_for(scan: Path, input_path: Path, output_root: Path) -> Path:
    """Mirror the input folder structure; a single-scan input maps to the output root."""
    if input_path.is_file():
        return output_root
    return output_root / scan.parent.relative_to(input_path)


def internal_scan_name(scan: Path, unique_id: str | None = None) -> str:
    """Filename used for the scan symlink inside its work dir.

    Scans already shaped "<id>_<lower|upper>" (or "STEM_<lower|upper>_<id>")
    are used unchanged, so tooth keys downstream stay clean and match what
    postprocess_predictions naturally produces (e.g. "118_lower_FDI_47").
    Otherwise the scan carries no id of its own (e.g. a generic "lower.stl"),
    so the id is taken from its parent directory instead — this project's
    usual layout is one folder per patient (e.g. ".../302/lower.stl" -> id
    "302"), which is also naturally unique, unlike the scan's own filename.
    `unique_id` is only a last-resort disambiguator for the rare case where
    two different parent folders share the same name within one batch.
    """
    if SCAN_NAME_RE.match(scan.stem):
        return scan.name
    arch = "lower" if "lower" in scan.stem else "upper"
    scan_id = scan.parent.name.replace("_", "-")
    if unique_id is not None:
        scan_id = f"{scan_id}-{unique_id}"
    return f"{scan_id}_{arch}{scan.suffix}"


class InferencePipeline:
    """Loads both models once (via the shared LandmarksPredictor engine) and
    runs the full pipeline scan by scan."""

    def __init__(self, args):
        self.engine = LandmarksPredictor(
            args.seg_config, args.seg_weight, args.bond_config, args.bond_weight,
            remesh=False,
            visualize_segmentation=args.vis_seg,
            visualize_segmentation_3d=args.vis_seg_3d,
            save_ply=args.save_ply,
            cache=TeethCache(),  # tooth meshes are always cached, never exported
            preprocessing=args.preprocessing,
            landmarks=args.landmarks,
            workers=args.workers,
        )
        self.workers = args.workers
        self.processing_time = 0.0  # cumulative segmentation + bond + postprocess time (excludes I/O)
        self.combined_landmarks: dict[str, dict] = {}  # tooth_key -> landmark data, flat like postprocess_predictions' own landmarks.json

    @property
    def cache(self) -> TeethCache:
        return self.engine.cache

    def run(self, scan: Path, out_dir: Path):
        """Segment one scan, predict its landmarks and export results to out_dir."""
        work_dir = Path(tempfile.mkdtemp(prefix=f"infer_{scan.stem}_"))
        try:
            # Fresh cache per scan: the cached datasets iterate over every cached scan.
            self.cache.clear()
            internal_name = internal_scan_name(scan)
            symlink_path = work_dir / internal_name
            symlink_path.symlink_to(scan.resolve())
            self.cache.preload_scan_mesh(symlink_path)

            proc_start = time.perf_counter()

            ok, msg = self.engine.run_segmentation(work_dir)
            if not ok:
                raise RuntimeError(msg)

            ok, msg = self.engine.run_bond_prediction(work_dir)
            if not ok:
                raise RuntimeError(msg)

            self.engine.postprocess(work_dir, visualize=False)

            self.processing_time += time.perf_counter() - proc_start

            landmarks_path = work_dir / "output_reg" / "results" / "landmarks.json"
            scan_landmarks = json.loads(landmarks_path.read_text()) if landmarks_path.exists() else {}
            self._record_landmarks(scan_landmarks)

            self._export(scan, internal_name, work_dir, out_dir, landmarks=scan_landmarks)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def run_batch(self, scans: list[Path], input_path: Path, output_root: Path) -> list[Path]:
        """Segment every scan in one pass, then bond every scan in one pass.

        Faster than calling run() per scan for large sets, since dataset/
        dataloader setup for segmentation and bonding is paid once instead
        of once per scan. Trade-off: all scans share one work dir, so a
        failure in the batched segmentation or bonding step aborts the
        whole batch instead of failing just one scan; only the later
        per-scan export step still fails scans individually.

        Returns the list of scans that failed.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="infer_batch_"))
        failed: list[Path] = []
        try:
            self.cache.clear()

            # Precompute internal names sequentially (cheap, no I/O). Ids come
            # from either the scan's own filename or its parent folder (see
            # internal_scan_name), both normally unique on their own; this is
            # just a last-resort safety net for the rare case where two scans
            # would still collide (e.g. same-named parent folders at
            # different paths under --input), disambiguated before anything
            # touches disk.
            seen_names: set[str] = set()
            planned = []  # (scan, internal_name)
            for i, scan in enumerate(scans):
                name = internal_scan_name(scan)
                if name in seen_names:
                    print(f"⚠️ Duplicate scan name '{name}' in this batch — disambiguating.")
                    name = internal_scan_name(scan, unique_id=f"{i:04d}")
                seen_names.add(name)
                planned.append((scan, name))

            def _preload_one(item):
                scan, internal_name = item
                symlink_path = work_dir / internal_name
                symlink_path.symlink_to(scan.resolve())
                self.cache.preload_scan_mesh(symlink_path)  # disk load + mesh cleanup, independent per scan
                return scan, internal_name, output_dir_for(scan, input_path, output_root)

            if self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    scan_entries = list(executor.map(_preload_one, planned))
            else:
                scan_entries = [_preload_one(item) for item in planned]

            proc_start = time.perf_counter()

            ok, msg = self.engine.run_segmentation(work_dir)
            if not ok:
                raise RuntimeError(f"batch segmentation failed: {msg}")

            ok, msg = self.engine.run_bond_prediction(work_dir)
            if not ok:
                raise RuntimeError(f"batch landmark prediction failed: {msg}")

            self.engine.postprocess(work_dir, visualize=False)

            self.processing_time += time.perf_counter() - proc_start

            landmarks_path = work_dir / "output_reg" / "results" / "landmarks.json"
            all_landmarks = json.loads(landmarks_path.read_text()) if landmarks_path.exists() else {}

            for scan, internal_name, out_dir in scan_entries:
                try:
                    prefix = f"{Path(internal_name).stem}_FDI_"
                    scan_landmarks = {k: v for k, v in all_landmarks.items() if k.startswith(prefix)}
                    self._record_landmarks(scan_landmarks)
                    self._export(scan, internal_name, work_dir, out_dir, landmarks=scan_landmarks)
                except Exception:
                    traceback.print_exc()
                    failed.append(scan)
        except Exception:
            traceback.print_exc()
            failed = list(scans)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        return failed

    def _record_landmarks(self, scan_landmarks: dict):
        """Accumulate this scan's landmarks into the run-wide combined output.

        Flat dict keyed by tooth_key, same shape postprocess_predictions
        itself writes per scan (e.g. "118_lower_FDI_47": {...}). Tooth keys
        are unique per scan (run_batch disambiguates any name collisions
        before this is ever called), so a plain update() is safe.
        """
        self.combined_landmarks.update(scan_landmarks)

    def write_combined_landmarks(self, output_root: Path) -> Path:
        """Write every processed scan's landmarks into one landmarks.json at output_root."""
        output_root.mkdir(parents=True, exist_ok=True)
        out_path = output_root / "landmarks.json"
        with open(out_path, "w") as f:
            json.dump(self.combined_landmarks, f, indent=4)
        print(f"\n💾 Saved combined landmarks for {len(self.combined_landmarks)} teeth to {out_path}")
        return out_path

    def _export(self, scan: Path, internal_name: str, work_dir: Path, out_dir: Path, landmarks: dict):
        """Copy the final predictions from the temporary work dir to out_dir.

        Internal pipeline artifacts are named after `internal_name` (the
        symlink used inside work_dir); exported files keep the user's
        original scan name.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        internal_stem = Path(internal_name).stem
        if self.engine.save_ply:
            landmarks_json_path = work_dir / f"{internal_stem}_landmarks.json"
            landmarks_json_path.write_text(json.dumps(landmarks))
            # One scan is one arch, and the exported name already carries it,
            # so keep a single file rather than the per-arch split.
            json_to_ply(landmarks_json_path, out_dir / f"{scan.stem}_landmarks.ply",
                        split_by_arch=False)
        shutil.copy(
            work_dir / "output_seg" / "result" / f"{internal_stem}_pred.npy",
            out_dir / f"{scan.stem}_seg.npy",
        )
        if self.engine.visualize_segmentation:
            views = work_dir / "output_seg" / f"{internal_stem}_segmentation_views.png"
            if views.exists():
                shutil.copy(views, out_dir / f"{scan.stem}_segmentation_views.png")
        if self.engine.visualize_segmentation_3d:
            seg_mesh = work_dir / "output_seg" / f"{internal_stem}_segmentation.ply"
            if seg_mesh.exists():
                shutil.copy(seg_mesh, out_dir / f"{scan.stem}_segmentation.ply")
        print(f"💾 Saved predictions for {scan.name} to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segments scans and predicts landmarks.")
    parser.add_argument("--input",         required=True, help="Scan file (.stl/.obj) or folder scanned recursively")
    parser.add_argument("--output",        required=True, help="Output folder (mirrors the input folder structure)")
    parser.add_argument("--seg-config",    required=True, help="Segmentation config file")
    parser.add_argument("--seg-weight",    required=True, help="Segmentation model weights")
    parser.add_argument("--bond-config",   required=True, help="Bond prediction config file")
    parser.add_argument("--bond-weight",   required=True, help="Bond prediction model weights")
    parser.add_argument("--preprocessing", help="YAML with per-arch scan normalization transforms")
    parser.add_argument("--vis-seg",       action="store_true", help="Also save a rendering of the segmentation")
    parser.add_argument("--vis-seg-3d",    action="store_true", help="Also save a 3D colored mesh of the segmentation (.ply)")
    parser.add_argument("--save-ply",      action="store_true", help="Saves landmarks as point cloud")
    parser.add_argument("--batch",         action="store_true",
                        help="Segment every scan in one pass, then bond every scan in one pass, "
                             "instead of processing scans one at a time. Faster for large scan "
                             "sets, but a failure in the batched segmentation/bonding step aborts "
                             "the whole batch instead of failing just one scan.")
    parser.add_argument("--landmarks",     nargs="+", choices=ALL_LANDMARKS, default=None,
                        help="Restrict landmark prediction/post-processing to these classes "
                             "(default: all). Skips the k-means/connected-components decoding "
                             "for unrequested classes, notably 'Planar' and 'Cusp'. "
                             "'Bracket', 'Incisal' and 'OuterPoint' are always computed since "
                             "every landmark's basePlane is defined relative to them.")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of worker threads for the CPU/IO-bound steps that don't "
                             "run on the GPU: loading scans from disk, splitting a segmented "
                             "scan into per-tooth meshes, and turning each tooth's predicted "
                             "heatmap into final landmark coordinates. Does not affect model "
                             "inference itself. Default: 1 (sequential).")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output)

    scans = []
    for scan in collect_scans(input_path):
        if "lower" in scan.stem or "upper" in scan.stem:
            scans.append(scan)
        else:
            print(f"⚠️ Skipping {scan}: filename must contain 'lower' or 'upper'")
    if not scans:
        sys.exit("❌ No valid scans found.")
    print(f"Found {len(scans)} scan(s) to process.")

    pipeline = InferencePipeline(args)
    loop_start = time.perf_counter()
    if args.batch:
        print(f"\n🦷 Processing {len(scans)} scan(s) in batch mode "
              f"(all segmentations, then all bondings)...")
        failed = pipeline.run_batch(scans, input_path, output_root)
    else:
        failed = []
        for i, scan in enumerate(scans, 1):
            print(f"\n{'=' * 70}\n🦷 [{i}/{len(scans)}] {scan.name}\n{'=' * 70}")
            try:
                pipeline.run(scan, output_dir_for(scan, input_path, output_root))
            except Exception:
                traceback.print_exc()
                failed.append(scan)
    total_with_io = time.perf_counter() - loop_start

    print(f"\n✅ Done: {len(scans) - len(failed)}/{len(scans)} scans processed.")
    for scan in failed:
        print(f"  ❌ Failed: {scan}")

    pipeline.write_combined_landmarks(output_root)

    succeeded = len(scans) - len(failed)
    print(f"\n⏱️  Total time — with I/O:    {total_with_io:.2f}s for {len(scans)} scan(s) "
          f"({total_with_io / len(scans):.2f}s/scan)")
    if succeeded:
        print(f"⏱️  Total time — without I/O: {pipeline.processing_time:.2f}s for {succeeded} scan(s) "
              f"({pipeline.processing_time / succeeded:.2f}s/scan)")


if __name__ == "__main__":
    main()
