import os
import sys
from typing import Iterator
import uuid
from datetime import datetime
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import debugpy
from pathlib import Path
import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from application.bond import ALL_LANDMARKS
from application.utils import teethland_output, write_rows
from application.timing import *
from application.cache import TeethCache
from application.pipeline import LandmarksPredictor
import shutil
import tempfile

def remove_temp_folder(temp_dir:Path):
    try:
        shutil.rmtree(str(temp_dir))
        print(f"Removed temporary folder {temp_dir}")
    except Exception as e:
        print(f"⚠️ Could not remove temporary folder {temp_dir}: {e}")

def get_samples(L:list[Path]) -> list[str]:
    all_samples = []
    for file in L:
        with open(file) as f:
            samples = f.read().splitlines()
            all_samples += samples
    return all_samples

def _merge_gold(kpt_path: Path, merged_gold: dict):
    patient_id = kpt_path.stem.replace("__kpt", "")
    with open(kpt_path, "r") as f:
        data = json.load(f)
    for obj in data["objects"]:
        merged_gold[obj["class"]][patient_id].append(obj["coord"])


def _setup_exp_dir(output_folder: Path, model: LandmarksPredictor) -> Path:
    """Create a timestamped experiment directory and copy model configs into it."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = output_folder / f"{timestamp}_{uuid.uuid4().hex[:6]}"
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(model.bond_config, exp_dir)
    shutil.copy(model.seg_config, exp_dir)
    return exp_dir


def _iter_patient_files(
    dataset_path: Path,
    test_samples: set[str],
    collect_gt: bool,
) -> Iterator[tuple[Path, Path, Path | None]]:
    """
    Yield (patient_dir, mesh_filepath, kpt_path_or_None) for each matching patient.
    Raises FileNotFoundError if collect_gt is True and a GT file is missing.
    """
    for arch in ("lower", "upper"):
        dirpath = dataset_path / arch
        if not dirpath.exists():
            continue
        for patient in (d for d in os.listdir(dirpath) if os.path.isdir(dirpath / d)):
            base = dirpath / patient / f"{patient}_{arch}"
            filepath = next(
                (base.with_suffix(ext) for ext in (".obj", ".stl") if base.with_suffix(ext).exists()),
                None,
            )
            if not filepath or f"{patient}_{arch}" not in test_samples:
                continue
            kpt_path = None
            if collect_gt:
                kpt_path = dirpath / patient / f"{patient}_{arch}__kpt.json"
                if not kpt_path.exists():
                    raise FileNotFoundError(
                        f"GT file missing for {patient}_{arch}. Disable --collect-gt or ensure GT files exist."
                    )
            yield dirpath / patient, filepath, kpt_path

def _save_gold(merged_gold: dict, output_folder: Path) -> None:
    """Serialize the merged gold standard dict to a pickle file."""
    final_gold = {cls: dict(patients) for cls, patients in merged_gold.items()}
    out_path = output_folder / "gold_standard.pkl"
    print("Saving GT pickle to {}".format(out_path))
    with open(out_path, "wb") as f:
        pickle.dump(final_gold, f)


@timed
def _run_prediction_timed(model: LandmarksPredictor, temp_dir: Path) -> None:
    """Measure processing only (segmentation + bonding + postprocessing)."""
    model.predict(temp_dir, clean_previous=False, postprocess=True)

def test_3dteethland_singles(
    dataset_path: Path,
    files: list[str],
    model: LandmarksPredictor,
    output_folder: Path,
    collect_gt: bool = False,
):
    exp_dir = _setup_exp_dir(output_folder, model)
    merged_gold = defaultdict(lambda: defaultdict(list))
    rows = []

    for patient_dir, _, kpt_path in _iter_patient_files(dataset_path, set(files), collect_gt):
        model.predict(patient_dir, clean_previous=True, postprocess=True)
        rows += teethland_output(patient_dir / "output_reg" / "results" / "landmarks.json")
        if kpt_path:
            _merge_gold(kpt_path, merged_gold)

    write_rows(rows, exp_dir / "predictions.csv")
    if collect_gt:
        _save_gold(merged_gold, output_folder)

def test_3dteethland_optimized(
    dataset_path: Path,
    model: LandmarksPredictor,
    files: list[str],
    output_folder: Path,
    collect_gt: bool = False,
):
    exp_dir = _setup_exp_dir(output_folder, model)
    temp_dir = Path(tempfile.mkdtemp(prefix="scans_", dir=exp_dir))
    print(f"Created temporary scans folder: {temp_dir}")
    merged_gold = defaultdict(lambda: defaultdict(list))
    print(f"Pre-loading scans...")

    def _preload_one(entry):
        _, filepath, kpt_path = entry
        if model.cache:
            model.cache.preload_scan_mesh(filepath)  # disk load + mesh cleanup, independent per scan
        (temp_dir / filepath.name).symlink_to(filepath)
        if kpt_path:
            _merge_gold(kpt_path, merged_gold)

    entries = list(_iter_patient_files(dataset_path, set(files), collect_gt))
    if model.workers > 1:
        with ThreadPoolExecutor(max_workers=model.workers) as executor:
            list(executor.map(_preload_one, entries))
    else:
        for entry in entries:
            _preload_one(entry)

    _run_prediction_timed(model, temp_dir)
    rows = teethland_output(temp_dir / "output_reg" / "results" / "landmarks.json")
    write_rows(rows, exp_dir / "predictions.csv")
    if collect_gt:
        _save_gold(merged_gold, output_folder)

parser = argparse.ArgumentParser(
    description="Segments and predicts landmarks on a oriented scan."
)
parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
# ===================== DATA PATHS ========================
parser.add_argument("--samples",        nargs="+", help="paths to files containing testing filenames")
parser.add_argument("--data-folder",    help="Absolute path of the data folder")
parser.add_argument("--output-folder",  help="Absolute path of the output folder where predictions will be saved.")
# ============= MODEL WEIGHTS AND CONFIGS =================
parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")
# ================== OPTIONALS =============================
parser.add_argument("--remesh",         required=False, action="store_true", help="Enables remeshing of scans")
parser.add_argument("--preprocessing",  required=False, help="Preprocessing")
parser.add_argument("--vis-seg",        required=False, action="store_true", help="Renders the segmentation as a flat PNG (3 viewpoints)")
parser.add_argument("--vis-seg-3d",     required=False, action="store_true", help="Saves the colored segmentation mesh as a 3D .ply file")
parser.add_argument("--save-ply",       required=False, action="store_true", help="Saves landmarks as point cloud")
parser.add_argument("--cache",          required=False, action="store_true", help="Cache teeth meshes in memory")
parser.add_argument("--collect-gt",     required=False, action="store_true", help="Looks for __kpt.json files and stores them in a pickle object.")
parser.add_argument("--landmarks",      required=False, nargs="+", choices=ALL_LANDMARKS, default=None,
                     help="Restrict landmark prediction/post-processing to these classes "
                          "(default: all). Skips the k-means/connected-components decoding "
                          "for unrequested classes, notably 'Planar' and 'Cusp'. "
                          "'Bracket', 'Incisal' and 'OuterPoint' are always computed since "
                          "every landmark's basePlane is defined relative to them.")
parser.add_argument("--workers",        required=False, type=int, default=1,
                     help="Number of worker threads for the CPU/IO-bound steps that don't run "
                          "on the GPU: loading scans from disk, splitting a segmented scan into "
                          "per-tooth meshes, and turning each tooth's predicted heatmap into "
                          "final landmark coordinates. Does not affect model inference itself. "
                          "Default: 1 (sequential).")

args = parser.parse_args()
if args.debug:
    debugpy.listen(("0.0.0.0", 5681))
    print(">>> Waiting for debugger on port 5681 …")
    debugpy.wait_for_client()
    print(">>> Debugger attached.")

cache = TeethCache() if args.cache else None
if cache: print("💾 Teeth caching enabled\n")
flattened = "/homes/mlugli/BracketPrediction/Teeth3DS/original_test_set_flattened"
all_files = get_samples(args.samples)
model = LandmarksPredictor(args.seg_config,
                            args.seg_weight,
                            args.bond_config,
                            args.bond_weight,
                            args.remesh,
                            args.vis_seg,
                            args.save_ply,
                            visualize_segmentation_3d=args.vis_seg_3d,
                            cache=cache,
                            preprocessing=args.preprocessing,
                            landmarks=args.landmarks,
                            workers=args.workers,
                            )

test_3dteethland_optimized(
    Path(args.data_folder),
    model,all_files,
    Path(args.output_folder),
    args.collect_gt)
timings.report()