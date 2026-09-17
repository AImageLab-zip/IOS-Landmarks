"""
Automated dental scan processing monitor.

Watches --data-root for patient folders. If it finds raw data in a patient
folder (a raw_data/ subdirectory with STEM_*.stl scans + a config_*.json
metadata file):
1) Applies scanTransformMatrix (found in the metadata json) to the mesh;
2) Rotates the scan by 180 degrees around Y axis;
3) Rotates the scan by 90 degrees around X axis;
4) Shifts the scan towards the center of mass and saves the offset in a file;
Then automatically runs segmentation and landmark prediction (bonding) via
the shared LandmarksPredictor engine (application/pipeline.py) — the same
engine main.py and infer.py use, so this script is only responsible for the
folder-watching / job-tracking / API-notification loop around it, not for
re-implementing the pipeline itself.

Tracks individual files to process new files added to existing patient
folders. Output layout, processing_status.json schema and the notify_api
call are unchanged from the previous production version.

Usage:
    python application/monitor.py \
        --data-root /workspace/application/data/ \
        --seg-config /workspace/application/app_configs/Pt_semseg_teeth3ds_app.py \
        --seg-weight /workspace/application/weights/segmentator_best.pth \
        --bond-config /workspace/application/app_configs/Pt_landmarks_app.py \
        --bond-weight /workspace/application/weights/heatmap_landmarks.pth \
        --check-interval 10

Every flag exposed by main.py's LandmarksPredictor (--remesh, --cache,
--landmarks, --workers, --save-ply, --vis-seg, --preprocessing) is also
available here; see --help.
"""
# CRITICAL: Set rendering environment variables BEFORE any imports that use VTK/graphics
import os
import sys
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"

import time
import argparse
from pathlib import Path
from datetime import datetime

import debugpy
import requests

from application.preprocessor import Preprocessor
from application.pipeline import LandmarksPredictor
from application.bond import ALL_LANDMARKS
from application.cache import TeethCache
from application.visualizers import plot_jaw
from application.utils import load_json, save_json

# Job status codes — kept in sync with the remote API
PENDING    = 0
PROCESSING = 1
COMPLETED  = 2
FAILED     = 3

class ScanMonitor:
    def __init__(
        self,
        data_root: Path,
        seg_config: Path,
        seg_weight: Path,
        bond_config: Path,
        bond_weight: Path,
        check_interval: int = 10,
        status_file: str = "processing_status.json",
        remesh: bool = False,
        cache: bool = False,
        landmarks: list[str] | None = None,
        workers: int = 1,
        save_ply: bool = False,
        vis_seg: bool = False,
        vis_seg_3d: bool = False,
        preprocessing: str | None = None,
    ):
        self.data_root      = Path(data_root)
        self.check_interval = check_interval
        self.status_file    = self.data_root / status_file
        self.prep           = Preprocessor()

        # Validate required paths
        for label, p in [
            ("Data root",            Path(data_root)),
            ("Segmentation config",  Path(seg_config)),
            ("Segmentation weights", Path(seg_weight)),
            ("Bond config",          Path(bond_config)),
            ("Bond weights",         Path(bond_weight)),
        ]:
            if not p.exists():
                raise ValueError(f"{label} does not exist: {p}")

        # Engine: loads both models once at startup, shared with main.py/infer.py
        self.engine = LandmarksPredictor(
            seg_config, seg_weight, bond_config, bond_weight,
            remesh=remesh,
            visualize_segmentation=vis_seg,
            visualize_segmentation_3d=vis_seg_3d,
            save_ply=save_ply,
            cache=TeethCache() if cache else None,
            preprocessing=preprocessing,
            landmarks=landmarks,
            workers=workers,
        )

        print(f"✅ Monitor initialised")
        print(f"   Data root     : {self.data_root}")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Status file   : {self.status_file}")
        print(f"   Remesh        : {remesh}")
        print(f"   Cache         : {cache}")
        print(f"   Workers       : {workers}")
        print(f"   Save PLY      : {save_ply}")
        print(f"   Vis. seg.     : {vis_seg}")
        print(f"   Vis. seg. 3D  : {vis_seg_3d}")
        print(f"   Landmarks     : {landmarks or 'all'}")
        print(f"   Preprocessing : {preprocessing or 'none (identity)'}")

    # ------------------------------------------------------------------
    # Status / API helpers
    # ------------------------------------------------------------------

    def load_status(self) -> dict:
        return load_json(self.status_file)

    def save_status(self, status: dict):
        save_json(self.status_file, status)

    def notify_api(self, job_id: str, status: int, message: str = "") -> bool:
        try:
            url     = f"https://autobonding.ing.unimore.it/api/update/{job_id}/"
            headers = {
                "Authorization": f"Bearer {os.getenv('API_TOKEN')}",
                "Content-Type":  "application/json",
            }
            resp = requests.post(url, json={"status": status, "logs": message},
                                 headers=headers, timeout=5)
            resp.raise_for_status()
            print(f"   ✅ API notified: status={status}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  API notification failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_pending_patients(self) -> list[Path]:
        """Return patient dirs that have work to do."""
        pending = []
        for item in sorted(self.data_root.iterdir()):
            if not item.is_dir():
                continue
            has_raw  = (item / "raw_data").is_dir()
            has_stls = any(item.glob("*.stl"))
            if has_raw or has_stls:
                pending.append(item)
        return pending

    def has_new_raw_files(self, patient_dir: Path) -> bool:
        raw_dir = patient_dir / "raw_data"
        if not raw_dir.is_dir():
            return False
        for raw_stl in raw_dir.glob("[sS][tT][eE][mM]_*.[sS][tT][lL]"):
            if not (patient_dir / raw_stl.name).exists():
                return True
        return False

    def unprocessed_stls(self, patient_dir: Path, patient_status: dict) -> list[str]:
        """STL files in patient_dir not yet marked processed or failed."""
        all_stls      = {f.name for f in sorted(patient_dir.glob("*.stl"))}
        done          = set(patient_status.get("processed_files", []))
        failed        = set(patient_status.get("failed_files", []))
        return sorted(all_stls - done - failed)

    def is_already_running(self, patient_status: dict) -> bool:
        history = patient_status.get("processing_history", [])
        if not history:
            return False
        last = history[-1]
        return (last.get("status") == "processing"
                and "completed_at" not in last
                and "failed_at"    not in last)

    def needs_processing(self, patient_dir: Path, patient_status: dict) -> bool:
        if self.is_already_running(patient_status):
            print(f"  ⏳ {patient_dir.name}: already running, skipping")
            return False
        if self.has_new_raw_files(patient_dir):
            return True
        return len(self.unprocessed_stls(patient_dir, patient_status)) > 0

    # ------------------------------------------------------------------
    # Post-run visualisation
    # ------------------------------------------------------------------

    def make_plots(self, patient_dir: Path):
        """Generate all visualisations. Runs after the API has been notified,
        so a slow render never delays the "done" notification."""
        try:
            self.engine.postprocess(patient_dir, visualize=True)
            print("✅ Per-tooth visualisations done")
        except Exception as e:
            print(f"⚠️  Per-tooth visualisation failed: {e}")

        try:
            plot_jaw(patient_dir, raw_scan=True)
            print("✅ Jaw visualisation done")
        except Exception as e:
            print(f"⚠️  Jaw visualisation failed: {e}")

    # ------------------------------------------------------------------
    # Main patient processor
    # ------------------------------------------------------------------

    def process_patient(self, patient_dir: Path, status: dict):
        patient_id = patient_dir.name

        # Ensure the patient has a status entry
        patient_status = status.setdefault(patient_id, {
            "processed_files": [],
            "failed_files":    [],
            "processing_history": [],
        })

        # ── Stage 1: pre-process raw scans if needed ──────────────────
        if self.has_new_raw_files(patient_dir):
            ok, handled = self.prep.preprocess_raw_scans(patient_id, patient_dir)
            if not ok:
                patient_status["processing_history"].append({
                    "started_at":  datetime.now().isoformat(),
                    "status":      "failed",
                    "failed_at":   datetime.now().isoformat(),
                    "error":       "Pre-processing raw scans failed",
                    "files":       handled,
                })
                patient_status["failed_files"] = list(
                    set(patient_status["failed_files"]) | set(handled)
                )
                self.save_status(status)
                print(f"❌ Pre-processing failed for {patient_id}. Aborting.")
                return

        # ── Stage 2: pick up unprocessed STL files ────────────────────
        todo = self.unprocessed_stls(patient_dir, patient_status)
        if not todo:
            print(f"  ℹ️  Nothing new to process for {patient_id}")
            return

        # All input must arrive via raw_data/ + config_<id>.json: the standard
        # orientation is now applied through preprocessing/production_preprocessing.yaml, which
        # would mis-rotate a scan dropped straight in already oriented. Fail such
        # patients explicitly rather than segmenting a mis-oriented scan.
        if not (patient_dir / "raw_data").is_dir():
            msg = (f"{patient_id}: loose STL(s) {todo} with no raw_data/ directory. "
                   f"Provide raw_data/STEM_*.stl + config_<id>.json.")
            print(f"❌ {msg}")
            patient_status["processing_history"].append({
                "started_at": datetime.now().isoformat(),
                "status":     "failed",
                "failed_at":  datetime.now().isoformat(),
                "error":      "Missing raw_data/ directory",
                "files":      todo,
            })
            patient_status["failed_files"] = list(set(patient_status["failed_files"]) | set(todo))
            self.save_status(status)
            return

        print(f"\n{'#'*70}")
        print(f"# Patient : {patient_id}")
        print(f"# Files   : {todo}")
        print(f"{'#'*70}")

        # Record start
        entry = {
            "started_at": datetime.now().isoformat(),
            "files":      todo,
            "status":     "processing",
        }
        patient_status["processing_history"].append(entry)
        self.save_status(status)

        def fail(reason: str, message: str):
            self.notify_api(patient_id, FAILED, message)
            entry.update({"status": "failed", "failed_at": datetime.now().isoformat(), "error": reason})
            patient_status["failed_files"] = list(set(patient_status["failed_files"]) | set(todo))
            self.save_status(status)

        self.engine.clear_cache()

        # ── Segmentation ──────────────────────────────────────────────
        ok, msg = self.engine.run_segmentation(patient_dir)
        if not ok:
            fail("Segmentation failed", msg)
            return

        # ── Bond prediction ───────────────────────────────────────────
        ok, msg = self.engine.run_bond_prediction(patient_dir)
        if not ok:
            fail("Bond prediction failed", msg)
            return

        try:
            self.engine.postprocess(patient_dir, visualize=False)
            self.engine.export_ply(patient_dir)
            print("✅ Results saved")
        except Exception as e:
            fail("Post-processing failed", str(e))
            return

        self.notify_api(patient_id, COMPLETED, "Processing completed!")
        self.make_plots(patient_dir)
        entry.update({"status": "completed", "completed_at": datetime.now().isoformat()})
        patient_status["processed_files"] = list(set(patient_status["processed_files"]) | set(todo))
        self.save_status(status)

        print(f"\n{'='*70}")
        print(f"✅ DONE — {patient_id}  |  files: {todo}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(f"\n{'='*70}")
        print("🔍 Dental Scan Monitor — running")
        print(f"{'='*70}\n")

        status = {}
        try:
            while True:
                # Re-read status each iteration (picks up manual edits)
                status = self.load_status()

                for patient_dir in self.find_pending_patients():
                    patient_status = status.get(patient_dir.name, {})
                    if not self.needs_processing(patient_dir, patient_status):
                        continue

                    print(f"  ▶ {patient_dir.name}: new work found")
                    self.notify_api(patient_dir.name, PROCESSING, "Processing")
                    self.process_patient(patient_dir, status)

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            n_ok   = sum(len(v.get("processed_files", [])) for v in status.values())
            n_fail = sum(len(v.get("failed_files",    [])) for v in status.values())
            print(f"\n⚠️  Stopped by user  |  processed={n_ok}  failed={n_fail}  patients={len(status)}")
        except Exception as e:
            print(f"\n❌ Monitor crashed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description="Monitor and automatically process dental scan directories"
    )
    parser.add_argument("--data-root",      required=True, help="Root directory with patient folders")
    parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
    parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
    parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
    parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")
    parser.add_argument("--check-interval", type=int, default=10, help="Seconds between scans (default: 10)")
    parser.add_argument("--status-file",    default="processing_status.json", help="Status filename")
    parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
    # ================== Same optionals as main.py / infer.py ==========
    parser.add_argument("--remesh",         action="store_true", help="Enables remeshing of scans")
    parser.add_argument("--preprocessing",  required=False, default=None,
                         help="YAML with per-arch scan normalization transforms, applied on top of "
                              "the raw-scan ingestion step. Omit for identity (matches old production).")
    parser.add_argument("--vis-seg",        action="store_true", help="Renders the segmentation as a flat PNG (3 viewpoints)")
    parser.add_argument("--vis-seg-3d",     action="store_true", help="Saves the colored segmentation mesh as a 3D .ply file")
    parser.add_argument("--save-ply",       action="store_true", help="Saves landmarks as point cloud")
    parser.add_argument("--cache",          action="store_true",
                         help="Cache teeth meshes in memory instead of writing them to disk under "
                              "output_seg/teeth/. Faster, but skips writing those per-tooth files — "
                              "leave disabled to keep the old production on-disk layout.")
    parser.add_argument("--landmarks",      nargs="+", choices=ALL_LANDMARKS, default=None,
                         help="Restrict landmark prediction/post-processing to these classes "
                              "(default: all). 'Bracket', 'Incisal' and 'OuterPoint' are always "
                              "computed since every landmark's basePlane is defined relative to them.")
    parser.add_argument("--workers",        type=int, default=1,
                         help="Number of worker threads for the CPU/IO-bound pipeline steps. "
                              "Does not affect model inference itself. Default: 1 (sequential).")
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Waiting for debugger on port 5681 …")
        debugpy.wait_for_client()
        print(">>> Debugger attached.")

    monitor = ScanMonitor(
        data_root=args.data_root,
        seg_config=args.seg_config,
        seg_weight=args.seg_weight,
        bond_config=args.bond_config,
        bond_weight=args.bond_weight,
        check_interval=args.check_interval,
        status_file=args.status_file,
        remesh=args.remesh,
        cache=args.cache,
        landmarks=args.landmarks,
        workers=args.workers,
        save_ply=args.save_ply,
        vis_seg=args.vis_seg,
        vis_seg_3d=args.vis_seg_3d,
        preprocessing=args.preprocessing,
    )
    monitor.run()


if __name__ == "__main__":
    main()
