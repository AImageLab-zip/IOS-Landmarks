"""
Shared inference engine: loads both models once and runs the full
segmentation + landmark-prediction pipeline on a single scan directory.

This is the "single entry point" for the segmentation + landmark procedure.
main.py (3DTeethLand benchmark harness) and application/monitor.py (the
production folder-watching service) both drive the pipeline through this one
class instead of each re-implementing model loading / stage orchestration.
"""
from pathlib import Path
import shutil

from application.segment_scan import run_segmentation_with_model
from application.bond import run_bond_with_model, postprocess_predictions
from application.utils import load_model
from application.cache import TeethCache
from application.visualizers import json_to_ply
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer


class LandmarksPredictor:
    """Loads the segmentation + landmark models once and runs them on demand.

    Call `run_segmentation` / `run_bond_prediction` / `postprocess` directly
    for fine-grained control (e.g. incremental processing, as the monitor
    does), or `predict` for the common "do everything for this directory"
    case (used by main.py).
    """

    def __init__(
        self,
        seg_config: str,
        seg_weight: str,
        bond_config: str,
        bond_weight: str,
        remesh: bool,
        visualize_segmentation: bool,
        save_ply: bool,
        visualize_segmentation_3d: bool = False,
        cache: TeethCache | None = None,
        preprocessing=None,
        landmarks: list[str] | None = None,
        workers: int = 1,
    ):
        self.seg_config = Path(seg_config)
        self.seg_weight = Path(seg_weight)
        self.bond_config = Path(bond_config)
        self.bond_weight = Path(bond_weight)
        self.remesh = remesh
        self.visualize_segmentation = visualize_segmentation
        self.visualize_segmentation_3d = visualize_segmentation_3d
        self.save_ply = save_ply
        self.cache = cache
        self.preprocessing = preprocessing
        self.landmarks = landmarks
        self.workers = workers
        self.preprocessor = ScanNormalizer(self.preprocessing)
        print("\n🔄 Loading models on GPU …")
        self.seg_cfg, self.seg_model = load_model(self.seg_config, self.seg_weight)
        self.bond_cfg, self.bond_model = load_model(self.bond_config, self.bond_weight)
        print("✅ Both models ready.\n")

    def __del__(self):
        try:
            import torch
            del self.seg_model, self.bond_model
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _clean_outputs(self, directory: Path):
        for name in ("output_reg", "output_seg"):
            target = directory / name
            if target.exists() and target.is_dir():
                shutil.rmtree(target)

    def clear_cache(self):
        """Drop cached tooth meshes/transforms between patients. Keeps any
        preloaded full-scan meshes (see TeethCache.clear_teeth_data)."""
        if self.cache:
            self.cache.clear_teeth_data()

    def run_segmentation(self, patient_dir: Path) -> tuple[bool, str]:
        print(f"\n{'='*70}\n🦷 SEGMENTATION — {patient_dir.name}\n{'='*70}")
        try:
            ok = run_segmentation_with_model(
                cfg=self.seg_cfg,
                model=self.seg_model,
                data_folder=patient_dir,
                remesh=self.remesh,
                visualize=self.visualize_segmentation,
                visualize_3d=self.visualize_segmentation_3d,
                cache=self.cache,
                preprocessor=self.preprocessor,
                workers=self.workers,
            )
            msg = f"Segmentation {'completed' if ok else 'failed'} for {patient_dir.name}"
            return ok, msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def run_bond_prediction(self, patient_dir: Path) -> tuple[bool, str]:
        print(f"\n{'='*70}\n📍 BOND PREDICTION — {patient_dir.name}\n{'='*70}")
        try:
            ok = run_bond_with_model(
                cfg=self.bond_cfg,
                model=self.bond_model,
                data_folder=patient_dir,
                cache=self.cache,
                target_landmarks=self.landmarks,
            )
            msg = f"Bond prediction {'completed' if ok else 'failed'} for {patient_dir.name}"
            return ok, msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def postprocess(self, patient_dir: Path, visualize: bool = False):
        """Project predictions back onto the mesh and rotate them into the
        original scan frame. `visualize` also renders per-tooth plots."""
        postprocess_predictions(
            patient_dir,
            visualize=visualize,
            cache=self.cache,
            preprocessor=self.preprocessor,
            workers=self.workers,
        )

    def export_ply(self, patient_dir: Path):
        """Write a landmarks point cloud per arch next to the JSON results."""
        if not self.save_ply:
            return
        written = json_to_ply(
            patient_dir / "output_reg" / "results" / "landmarks.json",
            patient_dir / "output_reg" / "results" / "landmarks.ply",
        )
        for path in written:
            print(f"💾 Saved landmark point cloud to: {path}")

    def predict(self, directory: Path, clean_previous: bool = True, postprocess: bool = False):
        """Run the full pipeline (segmentation → bond prediction → optional
        postprocessing) on a directory containing one or more scans."""
        if clean_previous:
            self._clean_outputs(directory)
        self.clear_cache()
        ok, msg = self.run_segmentation(directory)
        if not ok:
            print("Segmentation failed {}".format(msg))
            return
        ok, msg = self.run_bond_prediction(directory)
        if not ok:
            print("Bond prediction failed {}".format(msg))
            return
        if not postprocess:
            return
        self.postprocess(directory, visualize=False)
        print("✅ Results saved")
        self.export_ply(directory)
