"""
This process segments the scan and extracts a mesh for each single tooth.
It operates on the lower scan in the standard alignment and centered in it's
center of mass, while the upper scan is also rotated around the Y axis of 180 degrees
so that tooth 48 is overlapped with lower's tooth 28.
This rotation will be reversed later to go back to the original reference frame.
"""
import os
import threading
import trimesh
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer
MAPPING = {
    1: 48, 2: 47, 3: 46,
    4: 45, 5: 44, 6: 43,
    7: 42, 8: 41, 9: 31,
    10: 32, 11: 33, 12: 34,
    13: 35, 14: 36, 15: 37,
    16: 38
}
from application.visualizers import create_segmentation_visualization
from pointcept.engines.defaults import default_setup
from pointcept.engines.test import TESTERS
from pathlib import Path
import numpy as np
import json
from scipy.spatial import cKDTree
from application.utils import *
from application.cache import TeethCache

# matplotlib's pyplot state is process-global and not thread-safe; serialize
# figure creation when postprocess_segmentation runs across worker threads.
_VIZ_LOCK = threading.Lock()


def normalize(points: np.ndarray, flip:bool=False) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rotate 180° around y-axis (for upper scan), then normalize points to unit sphere centered at origin.
    Returns: (normalized_points, translation, scale)

    In production, the model runs on a version of the upper scan that is rotated
    of an extra 180 degrees around the Y axis, such that the lower and upper jaws are
    "overlapped", not registered anymore. Since during training we don't make the model
    robust to this sort of flip, we need to save the tooth meshes oriented as the model
    is used to. The lower jaw scan is untached, so we don't apply the extra rotation.
    """
    if flip:
        rotation_matrix = rotation_180_y()
        rotated_points = points @ rotation_matrix.T
    else: rotated_points = points
    
    centroid = np.mean(rotated_points, axis=0)
    centered_points = rotated_points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    max_distance = np.max(distances)
    scale = 1.0 / max_distance if max_distance > 0 else 1.0
    normalized_points = centered_points * scale
    
    return normalized_points, centroid, scale

def compute_dilation_masks(mask: np.ndarray, vertices: np.ndarray) -> dict[int, np.ndarray]:
    tooth_labels = np.unique(mask)
    tooth_labels = tooth_labels[tooth_labels != 0]
    verts = vertices.astype(np.float32)

    result = {}
    for label in tooth_labels:
        tooth_verts = verts[mask == label]
        if len(tooth_verts) == 0:
            continue

        bbox_min = tooth_verts.min(axis=0)
        bbox_max = tooth_verts.max(axis=0)
        tooth_size = np.linalg.norm(bbox_max - bbox_min)

        dilation_radius = 0.05 * tooth_size  # tune this
        tree = cKDTree(tooth_verts)
        dists, _ = tree.query(verts, workers=-1)
        result[label] = dists <= dilation_radius

    return result

def _remove_small_labels(mask:np.ndarray, points:np.ndarray, min_fraction:float=0.4) -> np.ndarray:
    tooth_labels = np.unique(mask[mask != 0])
    if len(tooth_labels) == 0:
        return mask
    
    mean_tooth_size = np.mean([np.sum(mask == l) for l in tooth_labels])
    min_component_size = int(min_fraction * mean_tooth_size)
    
    # Identify labels to delete (too few points)
    labels_to_delete = set()
    for label in tooth_labels:
        if np.sum(mask == label) < min_component_size:
            labels_to_delete.add(label)
    
    if not labels_to_delete:
        return mask
    
    # Build KD-tree for all points with valid labels (not to be deleted)
    valid_labels = np.isin(mask, list(labels_to_delete), invert=True)
    valid_point_indices = np.where(valid_labels)[0]
    
    if len(valid_point_indices) == 0:
        return mask
    valid_points = points[valid_point_indices]
    tree = cKDTree(valid_points)
    new_mask = mask.copy()
    # Reassign noisy points to nearest neighbor with different label
    for label in labels_to_delete:
        noisy_point_indices = np.where(mask == label)[0]
        noisy_points = points[noisy_point_indices]
        
        # Find nearest neighbor in valid labels
        dists, neighbor_indices = tree.query(noisy_points, workers=-1)
        nearest_valid_indices = valid_point_indices[neighbor_indices]
        
        # Reassign to the label of nearest valid point
        new_labels = mask[nearest_valid_indices]
        new_mask[noisy_point_indices] = new_labels
 
    return new_mask

def _keep_largest_components(mask: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    AI generated function.
    For each tooth class, keep only the largest connected component (based on
    mesh connectivity) and reassign all smaller components to gum (label 0).
    Args:
        mask:  (n_points,)  label per vertex (0 = gum, >0 = tooth class)
        faces: (n_faces, 3) triangle indices into the vertex array
    Returns:
        Mask where each tooth label contains only its largest connected component.
    """
    # Build undirected edge list from triangles once
    edges = get_edges(faces)
    new_mask = mask.copy()

    for tooth_label in np.unique(mask[mask != 0]):
        tooth_indices = np.where(new_mask == tooth_label)[0]
        if len(tooth_indices) == 0:
            continue

        # Keep only edges internal to this tooth
        tooth_set = np.zeros(len(new_mask), dtype=bool)
        tooth_set[tooth_indices] = True
        tooth_edges = edges[tooth_set[edges[:, 0]] & tooth_set[edges[:, 1]]]
        if len(tooth_edges) == 0:
            continue

        # Re-index to compact range [0, n_tooth_points)
        local_idx = np.full(len(new_mask), -1, dtype=int)
        local_idx[tooth_indices] = np.arange(len(tooth_indices))
        local_edges = local_idx[tooth_edges]  # (n_edges, 2)

        n = len(tooth_indices)
        graph = csr_matrix(
            (np.ones(len(local_edges)), (local_edges[:, 0], local_edges[:, 1])),
            shape=(n, n)
        )

        n_components, component_ids = connected_components(graph, directed=False)

        if n_components == 1:
            continue

        largest = np.bincount(component_ids).argmax()
        new_mask[tooth_indices[component_ids != largest]] = 0

    return new_mask
   

def clean_segmentation_mask(mask:np.ndarray, points:np.ndarray, faces:np.ndarray, min_fraction=0.4) -> np.ndarray:
    """
    1) Remove small noise labels by reassigning them to nearest neighbor labels.
    2) Applies CCL and keeps only the largest connected component for each tooth.
    """
    # Compute mean tooth size (in points) across all labeled teeth
    mask = _remove_small_labels(mask, points, min_fraction)
    mask = _keep_largest_components(mask, faces)
    return mask

def _core_submesh(vertices: np.ndarray, faces: np.ndarray, core_mask: np.ndarray) -> trimesh.Trimesh | None:
    """Undilated tooth mesh: the core vertices and the faces fully inside them.

    Vertices are NOT re-normalized, they are a subset of the same normalized
    coordinates the dilated mesh is built from, so the translation/scaling
    stored in the tooth JSON (and hence the denorm_matrix in landmarks.json)
    apply unchanged to this mesh. Returns None when no face survives.
    """
    if faces.size == 0:
        return None
    face_mask = core_mask[faces].all(axis=1)
    if not face_mask.any():
        return None
    old_to_new = np.full(len(vertices), -1, dtype=int)
    core_idx = np.where(core_mask)[0]
    old_to_new[core_idx] = np.arange(len(core_idx))
    return trimesh.Trimesh(vertices=vertices[core_mask],
                           faces=old_to_new[faces[face_mask]],
                           process=False)

def dilate_and_save_teeth(mask:np.ndarray, 
                          points:np.ndarray, 
                          faces:np.ndarray, 
                          base_name:str,
                          teeth_output_dir:Path,
                          cache:TeethCache = None,
                          core_output_dir:Path | None = None):
    """
    Splits the scan into one mesh per tooth. Two meshes are produced per tooth,
    both in the same normalized frame (see normalize()):
      * the dilated mesh (tooth + ~5% collar of gum / adjacent teeth), written
        to `teeth_output_dir`. This is what the landmark model was trained on
        and what bond.py consumes, so it stays the internal pipeline input;
      * the undilated mesh (only the vertices carrying the tooth's own label),
        written to `core_output_dir` (default: `<teeth_output_dir>_core`). This
        is the one exposed to external consumers; the denorm_matrix of the
        tooth applies to it as-is.
    """
    if core_output_dir is None:
        core_output_dir = teeth_output_dir.with_name(teeth_output_dir.name + "_core")
    if not cache:
        core_output_dir.mkdir(parents=True, exist_ok=True)
    unique_fdi_indices = np.unique(mask)
    print(f"Found {len(unique_fdi_indices)} unique classes: {unique_fdi_indices}")
 
    # Compute dilation masks for including gum around teeth
    dilation_masks = compute_dilation_masks(mask, points)

    # Process teeth 
    for fdi_index in unique_fdi_indices:
        if fdi_index == 0:
            continue  # Skip gum
 
        # Get points belonging to this tooth and nearby neighbors (gum or other teeth)
        # Use dilation mask directly to include all spatially close vertices
        combined_mask = dilation_masks[fdi_index]
        # Only keep largest connected component to avoid segmentation artifacts
        class_indices = np.where(combined_mask)[0]
        if len(class_indices) == 0:
            continue
 
        class_points = points[combined_mask]

        # Which of the exported vertices are the tooth itself rather than the
        # dilation collar. The collar is ~5% of the bbox diagonal (see
        # compute_dilation_masks) and on the mesial/distal sides it is made of
        # the *adjacent teeth*, which are in contact. Any measurement of this
        # tooth's own extent therefore has to be taken on the core alone.
        core_mask = (mask == fdi_index)[combined_mask]

        normalized_class_points, translation, scale = normalize(class_points, "upper" in base_name)

        # Stored as coordinates rather than an index mask: the tooth is written
        # out as STL, and reloading an STL re-indexes (and merges) vertices, so
        # a mask built here would not line up with the mesh bond.py reads back.
        core_points = normalized_class_points[core_mask]
 
        face_mask = np.all(np.isin(faces, class_indices), axis=1)
        class_faces_old_idx = faces[face_mask]
 
        # Remap face indices to new point array
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        class_faces = np.array([[old_to_new[idx] for idx in face] for face in class_faces_old_idx])

        # Create tooth mesh
        try:
            tooth_mesh = trimesh.Trimesh(
                vertices=normalized_class_points,
                faces=class_faces,
                process=False
            )
        except Exception:
            print(f"⚠️ Malformed tooth mesh for class {fdi_index}")
            continue  # fix: was missing, would crash on .save() below

        # Undilated mesh, same frame as tooth_mesh (no re-normalization).
        core_mesh = _core_submesh(normalized_class_points, class_faces, core_mask)
        if core_mesh is None:
            print(f"⚠️ No undilated faces for class {fdi_index}, core mesh skipped")

 
        # FDI index mapping
        if "lower" in base_name: fdi_index = MAPPING[fdi_index]
        if "upper" in base_name: fdi_index = MAPPING[fdi_index]-20
        
        # Prepare tooth key and transformation data
        tooth_key = f"{base_name}_FDI_{fdi_index}"
        json_data = {
            "translation": translation.tolist(),
            "scaling": float(scale)
        }
        
        # Store or save tooth
        if cache:
            # Cache mode: store in memory only
            cache.store_mesh(tooth_key, tooth_mesh, json_data)
            cache.store_core(tooth_key, core_points)
            if core_mesh is not None:
                cache.store_core_mesh(tooth_key, core_mesh)
            print(f"  Cached FDI {fdi_index}: {len(class_points)} points, {len(class_faces)} faces")
        else:
            # Disk mode: save to file
            stl_output_path = teeth_output_dir / f"{tooth_key}.stl"
            try:
                tooth_mesh.export(str(stl_output_path))
                # Save normalization parameters to JSON
                json_output_path = teeth_output_dir / f"{tooth_key}.json"
                with open(json_output_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                # Undilated vertices, consumed by the mesio-distal width
                # estimation in bond.py and by debug_shifted_landmarks.py.
                core_output_path = teeth_output_dir / f"{tooth_key}.core.npy"
                np.save(core_output_path, core_points)
                print(f"  Saved FDI {fdi_index}: {len(class_points)} points, {len(class_faces)} faces")
                print(f"    STL: {stl_output_path}")
                print(f"    JSON: {json_output_path}")
                print(f"    CORE: {core_output_path} ({len(core_points)}/{len(core_mask)} core verts)")
                # Undilated mesh: the one exposed externally. Kept out of
                # teeth_output_dir because the landmark dataset globs every
                # *.stl in there as a model input.
                if core_mesh is not None:
                    core_stl_output_path = core_output_dir / f"{tooth_key}.stl"
                    core_mesh.export(str(core_stl_output_path))
                    print(f"    CORE STL: {core_stl_output_path} ({len(core_mesh.faces)} faces)")
            except:
                print("⚠ Warning, could not export {}".format(stl_output_path))


def postprocess_segmentation(scan_file: Path,
                             mask_file: Path,
                             output_dir: Path,
                             remesh:bool =False,
                             visualize:bool = False,
                             visualize_3d:bool = False,
                             cache:TeethCache = None,
                             preprocessor:ScanNormalizer | None = None):
    """
    Postprocess segmentation results: split by tooth, normalize, and save.    
    Args:
        scan: Path to original STL file
        mask_file: Path to predicted segmentation mask (.npy)
        remesh: Enables remeshing of the scan
        output_dir: Output directory for processed teeth
    """
    
    print(f"Postprocessing {scan_file.name}...")
    arch = "lower" if "lower" in scan_file.name else "upper"
    # Load mesh and mask
    mesh = cache.get_scan_mesh(scan_file) if cache else None
    if mesh is None:
        mesh = trimesh.load_mesh(str(scan_file), process=False)
    if preprocessor: mesh = preprocessor.apply(mesh, arch) 
    mesh.merge_vertices()
    mask = np.load(mask_file) 
    if not is_consistent(mesh.vertices, mask): return
 
    base_name = scan_file.stem
    teeth_output_dir = output_dir / "teeth"

    teeth_output_dir.mkdir(parents=True, exist_ok=True)
    points = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    cleaned_mask = clean_segmentation_mask(mask, points, faces) 
    np.save(mask_file, cleaned_mask) # store cleaned segmentation mask

    if remesh:
        remeshed_teeth_output_dir = output_dir / "remeshed_teeth"
        remeshed_output_dir = output_dir / "remeshed"
        remeshed_scan_filename = output_dir / "remeshed" / Path(base_name).with_suffix(".stl")
        remeshed_output_dir.mkdir(parents=True, exist_ok=True)
        remeshed_teeth_output_dir.mkdir(parents=True, exist_ok=True)
        remeshed_scan = custom_remesh(scan_file) # run custom remeshing on full scan
        save_remeshed(remeshed_scan, remeshed_scan_filename)
        remeshed_scan_trimesh = trimesh.load_mesh(remeshed_scan_filename) # re-load using trimesh (fast)
        remeshed_mask = fit_segmask(cleaned_mask, points, remeshed_scan_trimesh.vertices)
        np.save(remeshed_output_dir / Path(base_name).with_suffix(".npy"), remeshed_mask)
    
        points_remeshed = np.array(remeshed_scan_trimesh.vertices)
        faces_remeshed = np.array(remeshed_scan_trimesh.faces)
        dilate_and_save_teeth(remeshed_mask.squeeze(), points_remeshed, faces_remeshed, base_name, remeshed_teeth_output_dir, cache=cache)

    if visualize or visualize_3d:
        try:
            with _VIZ_LOCK:
                # mesh/cleaned_mask are still in the preprocessing-normalized
                # frame (see `preprocessor.apply` above); undo that just for
                # the exported visualization so it lines up with the original
                # scan, without touching the frame used for teeth splitting.
                viz_mesh = preprocessor.apply_inverse(mesh, arch) if preprocessor else mesh
                create_segmentation_visualization(
                    viz_mesh, cleaned_mask, scan_file.stem, output_dir,
                    save_png=visualize, save_3d=visualize_3d,
                )
        except Exception as e: print(f"  ⚠️  Visualization failed (continuing anyway): {e}")
    # Get unique FDI indices from cleaned mask (excluding 0 which is gum)
    dilate_and_save_teeth(cleaned_mask, points, faces, base_name, teeth_output_dir, cache=cache)

@timed
def run_segmentation_with_model(cfg,
                                model,
                                data_folder: Path,
                                remesh=False,
                                visualize=False,
                                visualize_3d=False,
                                cache:TeethCache = None,
                                preprocessor:ScanNormalizer | None = None,
                                workers: int = 1) -> bool:
    """
    Run segmentation with a pre-loaded model.

    Args:
        cfg: Configuration object
        model: Pre-loaded segmentation model
        data_folder: Path to data folder containing STL files
        remesh: set to true if you want to save a remeshed version of the scan
        visualize: set to true if you want to render the segmentation as a flat PNG
        visualize_3d: set to true if you want to export the colored segmentation
            mesh as a 3D .ply file
        teethland: must set to true if testing on 3dteethland original scans
        cache: Optional TeethCache for mesh caching
        workers: number of scans to postprocess (mask cleanup + per-tooth mesh
            splitting) concurrently. 1 (default) processes scans sequentially.
            Uses a thread pool, not separate processes, so `cache` stays a
            single shared in-memory store visible to every scan.
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set data paths in config before setup
        cfg._cfg_dict["data_root"] = str(data_folder)
        cfg._cfg_dict["save_path"] = str(Path(data_folder) / "output_seg")
        cfg._cfg_dict["data"]["test"]["data_root"] = str(data_folder)
        cfg._cfg_dict["data"]["test"]["preprocessing"] = (
            str(preprocessor.config_path) if preprocessor and preprocessor.config_path else None
        )
 
        os.makedirs(cfg.save_path, exist_ok=True)
 
        # Set up configuration
        cfg = default_setup(cfg)
        if cache:
            cfg._cfg_dict["data"]["test"]["custom_cache"] = cache
            cfg._cfg_dict["data"]["test"]["type"] = "IosDatasetTeeth3dsCached"
 
        # Build and run tester with cached model
        test_cfg = dict(cfg=cfg, model=model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        tester.test()
 
        # Postprocessing: split and normalize teeth
        print("\n" + "="*80)
        print("Starting postprocessing...")
        print("="*80 + "\n")
 
        output_folder = Path(cfg.save_path)
 
        # Find STL files in data folder
        scans = list(data_folder.glob("*.stl")) + list(data_folder.glob("*.obj"))

        def _postprocess_one(scan):
            mask_file = output_folder / "result" / f"{scan.stem}_pred.npy"
            postprocess_segmentation(scan,
                mask_file,
                output_folder,
                remesh=remesh,
                visualize=visualize,
                visualize_3d=visualize_3d,
                cache=cache,
                preprocessor=preprocessor)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # list() drains the map so any worker exception is raised here.
                list(executor.map(_postprocess_one, scans))
        else:
            for scan in scans:
                _postprocess_one(scan)

        print("\n" + "="*80)
        print("Postprocessing complete!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"❌ Error in segmentation processing: {e}")
        import traceback
        traceback.print_exc()
        return False
