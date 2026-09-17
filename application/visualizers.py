# Create plane points
import matplotlib.pyplot as plt
from pathlib import Path
import json
import trimesh
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from plyfile import PlyData, PlyElement

# simple color map per landmark type
COLORS = {
    "bracket":  [255, 0, 0],
    "incisal":  [0, 255, 0],
    "outer":    [0, 0, 255],
    "gingival": [255, 255, 0],
    "mesial":   [255, 0, 255], #magenta
    "distal":   [0, 255, 255], #cyan
    "inner":    [128, 128, 255],
    "facial":   [255, 128, 0],
    "cusps":    [128, 255, 128],
    "planar":   [200, 200, 200],
    "boundary_mesial": [128, 0, 128],  # dark magenta, pairs with "mesial"
    "boundary_distal": [0, 128, 128],  # dark cyan, pairs with "distal"
}

# Entries in landmarks.json that are not point data and must never reach the
# point cloud: "basePlane" is a dict of axes, "denorm_matrix" a 4x4 transform.
PLY_SKIP_KEYS = {"basePlane", "gingival", "denorm_matrix", "denorm_matrix_tooth"}

AUTOBONDING_MAPPING = {
    48: 1, 47: 2, 46: 3,
    45: 4, 44: 5, 43: 6,
    42: 7, 41: 8, 31: 9,
    32: 10, 33: 11, 34: 12,
    35: 13, 36: 14, 37: 15,
    38: 16, 0:0
}
INVERSE_AUTOBONDING_MAPPING = {v: k for k, v in AUTOBONDING_MAPPING.items()}

def _marker_kwargs(style: dict) -> dict:
    """Scatter kwargs for one landmark style.

    Boundary points are drawn hollow and larger so they stay readable next to
    the prediction they were derived from, which shares their marker shape.
    """
    kw = dict(s=style.get('size', 100), marker=style['marker'],
              linewidths=style.get('lw', 2), zorder=5)
    if style.get('hollow'):
        kw['facecolors'] = 'none'
        kw['edgecolors'] = style['color']
    else:
        kw['c'] = style['color']
        kw['edgecolors'] = 'black'
    return kw


def plot_teeth(points_dict: dict, 
               v_io:np.ndarray, v_perp:np.ndarray,
               vertices:np.ndarray, 
               patient_id:str, fdi:int,
               output_dir:Path):

    fig, axes = plt.subplots(1,3,figsize=(18,6))
    plane_size = 0.5  # smaller
    plane_res = 30  # denser
    s = np.linspace(-plane_size, plane_size, plane_res)
    t = np.linspace(-plane_size, plane_size, plane_res)
    ss, tt = np.meshgrid(s, t)

    bracket = points_dict.get('Bracket')
    if bracket is None:
        print("No bracket point for plane")
        return
    plane_points = bracket + ss[..., None] * v_io + tt[..., None] * v_perp
    plane_points = plane_points.reshape(-1, 3)

    # Plot the plane on all three views
    axes[0].scatter(plane_points[:, 0], plane_points[:, 1], c='gray', s=5, alpha=0.5, label='Plane')
    axes[1].scatter(plane_points[:, 0], plane_points[:, 2], c='gray', s=5, alpha=0.5)
    axes[2].scatter(plane_points[:, 1], plane_points[:, 2], c='gray', s=5, alpha=0.5)

    # Define styles for each point type
    point_styles = {
        'Bracket': {'color': 'orange', 'marker': 'o', 'label': 'Bracket'},
        'Incisal': {'color': 'cyan', 'marker': 's', 'label': 'Incisal'},
        'Gingival': {'color': 'green', 'marker': 'D', 'label': 'Gingival'},
        'Mesial': {'color': 'purple', 'marker': 'v', 'label': 'Mesial'},
        'Distal': {'color': 'brown', 'marker': '^', 'label': 'Distal'},
        'BoundaryMesial': {'color': 'magenta', 'marker': 'v', 'label': 'Mesial (boundary)',
                           'hollow': True, 'size': 200, 'lw': 2.5},
        'BoundaryDistal': {'color': 'red', 'marker': '^', 'label': 'Distal (boundary)',
                           'hollow': True, 'size': 200, 'lw': 2.5},
        # Keys must match the names in bond.SINGLE_LANDMARKS ('InnerPoint' /
        # 'OuterPoint'); as plain 'Inner'/'Outer' they never matched and these
        # two landmarks were silently absent from every single-tooth plot.
        'InnerPoint': {'color': 'pink', 'marker': 'P', 'label': 'Inner'},
        'OuterPoint': {'color': 'teal', 'marker': '>', 'label': 'Outer'},
        'Planar': {'color': 'red', 'marker': '*', 'label': 'Planar'},
        'Cusp': {'color': 'blue', 'marker': 'X', 'label': 'Cusp'},
    }


    def draw_md_correction(ax, i, j):
        """Dashed leader from each prediction to its corrected boundary point,
        plus the chord whose length is the measured mesio-distal width."""
        bm, bd = points_dict.get('BoundaryMesial'), points_dict.get('BoundaryDistal')
        if bm is None or bd is None:
            return
        ax.plot([bm[i], bd[i]], [bm[j], bd[j]], c='red', lw=1.6, alpha=0.75,
                zorder=4, label='M-D width')
        first = True
        for pred_key, bnd in (('Mesial', bm), ('Distal', bd)):
            pred = points_dict.get(pred_key)
            if pred is None:
                continue
            ax.plot([pred[i], bnd[i]], [pred[j], bnd[j]], c='0.3', lw=1.2,
                    ls='--', alpha=0.9, zorder=4,
                    label='correction' if first else None)
            first = False

    # View 1: XY plane (looking down Z-axis)
    ax = axes[0]
    ax.scatter(vertices[:, 0], vertices[:, 1], c='lightblue', s=3, alpha=0.3, label='Mesh')
    for name, points in points_dict.items():
        if name in point_styles:
            style = point_styles[name]
            if isinstance(points, list):
                for p in points:
                    ax.scatter(p[0], p[1], label=style['label'], **_marker_kwargs(style))
                    style['label'] = None  # only label once
            else:
                ax.scatter(points[0], points[1], label=style['label'], **_marker_kwargs(style))
    draw_md_correction(ax, 0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY View (Top)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # View 2: XZ plane (looking from Y-axis)
    ax = axes[1]
    ax.scatter(vertices[:, 0], vertices[:, 2], c='lightblue', s=3, alpha=0.3, label='Mesh')
    for name, points in points_dict.items():
        if name in point_styles:
            style = point_styles[name]
            if isinstance(points, list):
                for p in points:
                    ax.scatter(p[0], p[2], label=style['label'], **_marker_kwargs(style))
                    style['label'] = None
            else:
                ax.scatter(points[0], points[2], label=style['label'], **_marker_kwargs(style))
    draw_md_correction(ax, 0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('XZ View (Front)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # View 3: YZ plane (looking from X-axis)
    ax = axes[2]
    ax.scatter(vertices[:, 1], vertices[:, 2], c='lightblue', s=3, alpha=0.3, label='Mesh')
    for name, points in points_dict.items():
        if name in point_styles:
            style = point_styles[name]
            if isinstance(points, list):
                for p in points:
                    ax.scatter(p[1], p[2], label=style['label'], **_marker_kwargs(style))
                    style['label'] = None
            else:
                ax.scatter(points[1], points[2], label=style['label'], **_marker_kwargs(style))
    draw_md_correction(ax, 1, 2)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('YZ View (Side)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.suptitle(f'Patient {patient_id} - Tooth FDI {fdi}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"patient_{patient_id}_FDI_{fdi}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved visualization: {output_file}")


def plot_jaw(data_folder:Path, raw_scan:bool = False):
    """
    Unified visualization for jaw predictions.

    If raw_scan is False:
      - Loads data_folder/output_reg/results/projected_points.json
      - Uses jaw meshes from data_folder (files like STEM_lower_0002.stl)
      - Groups teeth by jaw using tooth_key parsing (same as previous visualize_jaw_with_predictions)

    If raw_scan is True:
      - Loads data_folder/output_reg/results/projected_points_rotated.json
      - Uses raw scans from data_folder/raw_data (e.g. STEM_lower_0002.stl)
      - Applies scanTransformMatrix from config_<patient>.json to each raw scan before plotting
      - Groups rotated teeth by jaw based on presence of 'upper'/'lower' in tooth_key (similar to previous visualize_rotated_jaw_with_predictions)
    """
    if raw_scan:
        points_file = data_folder / "output_reg" / "results" / "projected_points_rotated.json"
    else:
        points_file = data_folder / "output_reg" / "results" / "projected_points.json"

    with open(points_file, 'r') as f: all_points = json.load(f)
    print(f"Loaded {'rotated' if raw_scan else 'original'} predictions for {len(all_points)} teeth")

    # Build jaw groups: mapping jaw_key -> list of (fdi, points_data)
    jaw_groups = {}
    if not raw_scan:
        # previous behavior: parse tooth_key format "STEM_lower_0002_FDI_47"
        for tooth_key, points_data in all_points.items():
            parts = tooth_key.split('_')
            try:
                jaw_type = 'lower' if 'lower' in parts else 'upper'
                patient_idx = parts.index(jaw_type)
                patient_id = parts[patient_idx + 1]
                fdi_idx = parts.index('FDI')
                fdi = parts[fdi_idx + 1]
                jaw_key = f"STEM_{jaw_type}_{patient_id}"
                jaw_groups.setdefault(jaw_key, []).append((fdi, points_data))
            except (ValueError, IndexError):
                print(f"⚠️ Could not parse tooth key: {tooth_key}")
                continue
    else:
        # rotated: group by raw scan files found in raw_data
        raw_data_dir = data_folder / "raw_data"
        if not raw_data_dir.exists():
            print(f"⚠️ raw_data directory not found: {raw_data_dir}")
            return
        scan_files = list(raw_data_dir.glob("*.stl"))
        if not scan_files:
            print(f"⚠️ No raw scans (.stl) found in {raw_data_dir}")
            return
        # For each scan file (jaw), collect teeth whose tooth_key contains 'upper'/'lower' matching this jaw
        for scan_file in scan_files:
            jaw_key = scan_file.stem  # e.g. STEM_lower_0002
            jaw_type = 'lower' if 'lower' in jaw_key else 'upper'
            jaw_groups[jaw_key] = []
            for tooth_key, points_data in all_points.items():
                if jaw_type in tooth_key:
                    fdi = tooth_key.split('FDI_')[-1]
                    jaw_groups[jaw_key].append((fdi, points_data))

    # Plot each jaw
    viz_dir = data_folder / "output_reg" / "jaw_plots"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for jaw_key, teeth_data in jaw_groups.items():
        if not teeth_data:
            continue

        # Determine jaw mesh path
        if raw_scan:
            scan_file = data_folder / "raw_data" / f"{jaw_key}.stl"
            patient_id = data_folder.name
            config_file = data_folder / "raw_data" / f"config_{patient_id}.json"
            transform_matrix = np.eye(4)
            with open(config_file, 'r') as f:
                cfg = json.load(f)
                transform_matrix = np.array(cfg['scanTransformMatrix']).reshape(4, 4)
                mesh = trimesh.load_mesh(scan_file)
                mesh.apply_transform(transform_matrix)

        else:
            jaw_mesh_file = data_folder / f"{jaw_key}.stl"
            mesh = trimesh.load_mesh(jaw_mesh_file)

        vertices = mesh.vertices
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        axes[0].scatter(vertices[:, 0], vertices[:, 1], c='lightgray', s=1, alpha=0.7, label='Jaw Mesh')
        axes[1].scatter(vertices[:, 0], vertices[:, 2], c='lightgray', s=1, alpha=0.7)
        axes[2].scatter(vertices[:, 1], vertices[:, 2], c='lightgray', s=1, alpha=0.7)

        colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(teeth_data)))
        legend_added = False  # Track if we've added point type legend
        for idx, (fdi, points_data) in enumerate(teeth_data):
            color = colors[idx]
            incisal = np.array(points_data.get('incisal', [0, 0, 0]))
            outer = np.array(points_data.get('outer', [0, 0, 0]))
            base_plane = points_data.get('basePlane', {})
            origin = np.array(base_plane.get('origin', [0, 0, 0]))
            x_axis = np.array(base_plane.get('xAxis', [0, 0, 0]))
            y_axis = np.array(base_plane.get('yAxis', [0, 0, 0]))

            # Plot bracket point (origin of the plane)
            label_bracket = 'Bracket' if not legend_added else None
            axes[0].scatter(origin[0], origin[1], c=[color], s=120, marker='o', edgecolors='black', linewidths=2, label=label_bracket, zorder=5)
            axes[1].scatter(origin[0], origin[2], c=[color], s=120, marker='o', edgecolors='black', linewidths=2, zorder=5)
            axes[2].scatter(origin[1], origin[2], c=[color], s=120, marker='o', edgecolors='black', linewidths=2, zorder=5)

            # Plot mesial and distal points if they exist
            if 'mesial' in points_data and points_data['mesial'] is not None:
                mesial_pt = np.array(points_data['mesial'])
                label_mesial = 'Mesial' if not legend_added else None
                axes[0].scatter(mesial_pt[0], mesial_pt[1], c=[color], s=80, marker='v', 
                               edgecolors='black', linewidths=1, alpha=0.7, label=label_mesial, zorder=4)
                axes[1].scatter(mesial_pt[0], mesial_pt[2], c=[color], s=80, marker='v', 
                               edgecolors='black', linewidths=1, alpha=0.7, zorder=4)
                axes[2].scatter(mesial_pt[1], mesial_pt[2], c=[color], s=80, marker='v', 
                               edgecolors='black', linewidths=1, alpha=0.7, zorder=4)
            
            if 'distal' in points_data and points_data['distal'] is not None:
                distal_pt = np.array(points_data['distal'])
                label_distal = 'Distal' if not legend_added else None
                axes[0].scatter(distal_pt[0], distal_pt[1], c=[color], s=80, marker='^', 
                               edgecolors='black', linewidths=1, alpha=0.7, label=label_distal, zorder=4)
                axes[1].scatter(distal_pt[0], distal_pt[2], c=[color], s=80, marker='^', 
                               edgecolors='black', linewidths=1, alpha=0.7, zorder=4)
                axes[2].scatter(distal_pt[1], distal_pt[2], c=[color], s=80, marker='^', 
                               edgecolors='black', linewidths=1, alpha=0.7, zorder=4)
            
            legend_added = True

            # plot base plane
            v_x = x_axis - origin
            v_y = y_axis - origin
            if np.linalg.norm(v_x) > 1e-6 and np.linalg.norm(v_y) > 1e-6:
                v_x = v_x / np.linalg.norm(v_x)
                v_y = v_y / np.linalg.norm(v_y)
                plane_size = 2.0
                plane_res = 20
                s = np.linspace(-plane_size, plane_size, plane_res)
                t = np.linspace(-plane_size, plane_size, plane_res)
                ss, tt = np.meshgrid(s, t)
                plane_points = origin + ss[..., None] * v_x + tt[..., None] * v_y
                plane_points = plane_points.reshape(-1, 3)
                axes[0].scatter(plane_points[:, 0], plane_points[:, 1], c=[color], s=2, alpha=0.2, zorder=3)
                axes[1].scatter(plane_points[:, 0], plane_points[:, 2], c=[color], s=2, alpha=0.2, zorder=3)
                axes[2].scatter(plane_points[:, 1], plane_points[:, 2], c=[color], s=2, alpha=0.2, zorder=3)

        # Configure plots
        axes[0].set_xlabel('X', fontsize=12)
        axes[0].set_ylabel('Y', fontsize=12)
        axes[0].set_title('XY View (Top)', fontsize=14, fontweight='bold')
        
        # Add legend for point types
        from matplotlib.lines import Line2D
        point_type_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
                   markeredgecolor='black', markeredgewidth=1.5, label='Bracket'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=7, 
                   markeredgecolor='black', markeredgewidth=1, label='Mesial'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=7, 
                   markeredgecolor='black', markeredgewidth=1, label='Distal'),
        ]
        legend1 = axes[0].legend(handles=point_type_elements, loc='upper left', fontsize=10, 
                                title='Point Types', title_fontsize=11, frameon=True)
        axes[0].add_artist(legend1)
        
        # Add legend for FDI numbers (tooth colors)
        fdi_elements = [plt.scatter([], [], c=[colors[idx]], s=100, marker='o', 
                                   edgecolors='black', linewidths=2, label=f'FDI {fdi}')
                       for idx, (fdi, _) in enumerate(teeth_data)]
        axes[0].legend(handles=fdi_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=8, title='Teeth', title_fontsize=9, frameon=True)
        
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')

        axes[1].set_xlabel('X', fontsize=12)
        axes[1].set_ylabel('Z', fontsize=12)
        axes[1].set_title('XZ View (Front)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')

        axes[2].set_xlabel('Y', fontsize=12)
        axes[2].set_ylabel('Z', fontsize=12)
        axes[2].set_title('YZ View (Side)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_aspect('equal', adjustable='box')

        plt.suptitle(f'{"Rotated " if raw_scan else ""}Predictions - {jaw_key}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        out_name = f"{jaw_key}_rotated_predictions.png" if raw_scan else f"{jaw_key}_predictions.png"
        output_file = viz_dir / out_name
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved visualization: {output_file}")

def create_segmentation_visualization(mesh:trimesh.Trimesh,
                                      mask:np.ndarray,
                                      name:str,
                                      output_dir: Path,
                                      save_png:bool = True,
                                      save_3d:bool = False):
    """
    Create a visualization of the segmented dental arch from 3 viewpoints.
    Uses a smooth gradient colormap with better distinguishability between teeth.

    save_png: render the usual 3-viewpoint flat image.
    save_3d: also export the per-vertex-colored mesh as a real 3D file
        (<name>_segmentation.ply) that can be opened/rotated in a viewer.
    """
    # Convert to pyvista mesh
    pv_mesh = pv.from_trimesh(mesh)
    if len(mask) != len(pv_mesh.points):
        print("Warning: Cannot visualize - mask length mismatch")
        return

    cmap_custom = LinearSegmentedColormap.from_list(
        "smooth_dental_gradient",
        [
            "#1E5BFF",  # blue
            "#00A9FF",  # sky blue
            "#00D4C7",  # cyan/teal
            "#38D66B",  # green
            "#DCEB00",  # yellow-green
            "#FFF066",  # soft yellow
        ],
        N=1024
    )
    # Assign colors based on FDI index
    unique_idx = np.unique(mask)
    non_zero_fdi = unique_idx[unique_idx != 0]
    n_teeth = len(non_zero_fdi)
    colors = np.zeros((len(mask), 3))
    color_map = {}
    # Use only the central part of the gradient to avoid overly dark/light extremes
    if n_teeth > 1:
        sampled_positions = np.linspace(0.08, 0.92, n_teeth)
    else:
        sampled_positions = np.array([0.5])

    for fdi_val in unique_idx:
        if fdi_val == 0:
            color = np.array([0.7, 0.7, 0.7])  # gum
            colors[mask == fdi_val] = color
            color_map[fdi_val] = color
        else:
            tooth_idx = np.where(non_zero_fdi == fdi_val)[0][0]
            rgb = np.array(cmap_custom(sampled_positions[tooth_idx])[:3])
            colors[mask == fdi_val] = rgb
            color_map[fdi_val] = rgb

    pv_mesh["colors"] = colors

    if save_png:
        # Define three viewpoints
        viewpoints = [
            {"azimuth": 0, "elevation": 0, "title": "Front"},
            {"azimuth": 90, "elevation": 0, "title": "Side"},
            {"azimuth": 0, "elevation": 90, "title": "Top"}
        ]

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 5))

        for idx, vp in enumerate(viewpoints):
            plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
            plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, lighting=False)
            plotter.camera.azimuth = vp["azimuth"]
            plotter.camera.elevation = vp["elevation"]
            plotter.camera.zoom(1.3)
            img = plotter.screenshot(return_img=True)
            plotter.close()

            ax = fig.add_subplot(1, 3, idx + 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(vp["title"], fontsize=14, fontweight="bold")

        # Add legend
        legend_elements = [
            Patch(
                facecolor=color_map[idx],
                label=f'FDI {INVERSE_AUTOBONDING_MAPPING[int(idx)] - 20 * ("upper" in name)}'
            )
            for idx in sorted(unique_idx)
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=min(8, len(unique_idx)),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.08)
        )

        plt.suptitle(f"Segmentation: {name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        vis_output_path = output_dir / f"{name}_segmentation_views.png"
        plt.savefig(vis_output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved visualization: {vis_output_path}")

    if save_3d:
        vertex_colors = np.zeros((len(colors), 4), dtype=np.uint8)
        vertex_colors[:, :3] = np.clip(colors * 255, 0, 255).astype(np.uint8)
        vertex_colors[:, 3] = 255
        colored_mesh = mesh.copy()
        colored_mesh.visual.vertex_colors = vertex_colors
        mesh_output_path = output_dir / f"{name}_segmentation.ply"
        colored_mesh.export(mesh_output_path)
        print(f"  Saved 3D segmentation mesh: {mesh_output_path}")


PLY_VERTEX_DTYPE = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]


def _tooth_points(tooth: dict) -> list:
    """Coloured landmark points for one tooth entry of landmarks.json."""
    out = []
    for key, value in tooth.items():

        if key in PLY_SKIP_KEYS or not value:
            continue  # axes, transforms and missing landmarks

        color = COLORS.get(key, [255, 255, 255])
        pts = value if isinstance(value[0], list) else [value]
        for pt in pts:
            # Guards against any future non-point entry sneaking in: the
            # vertex dtype has exactly three coordinate fields.
            if len(pt) != 3:
                continue
            out.append((*pt, *color))
    return out


def _write_ply(vertices: list, path: Path) -> Path:
    arr = np.array(vertices, dtype=PLY_VERTEX_DTYPE)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path


def json_to_ply(json_path, output_ply, split_by_arch: bool = True):
    """Write landmarks.json out as a coloured point cloud.

    With `split_by_arch` (the default) one file is written per arch, named
    after `output_ply`: landmarks.ply becomes landmarks_upper.ply and
    landmarks_lower.ply. The two arches are overlapped in the model's working
    frame, so a combined cloud is hard to read. An arch with no landmarks is
    not written at all.

    Pass split_by_arch=False when the JSON is known to hold a single arch and
    the caller has already named the file accordingly (see infer.py).

    Returns the list of paths actually written.
    """
    with open(json_path) as f:
        data = json.load(f)
    output_ply = Path(output_ply)

    if not split_by_arch:
        vertices = [v for tooth in data.values() for v in _tooth_points(tooth)]
        return [_write_ply(vertices, output_ply)] if vertices else []

    per_arch = {"upper": [], "lower": []}
    unknown = []
    for tooth_key, tooth in data.items():
        key_l = str(tooth_key).lower()
        # Both naming conventions carry the arch: "STEM_lower_<id>_FDI_47"
        # and "<id>_lower_FDI_47".
        arch = "lower" if "lower" in key_l else "upper" if "upper" in key_l else None
        if arch is None:
            unknown.append(tooth_key)
            continue
        per_arch[arch].extend(_tooth_points(tooth))

    if unknown:
        print(f"⚠️  {len(unknown)} tooth key(s) in {Path(json_path).name} name neither "
              f"arch and were left out of the PLY (e.g. {unknown[0]})")

    written = []
    for arch, vertices in per_arch.items():
        if not vertices:
            continue
        written.append(_write_ply(
            vertices,
            output_ply.with_name(f"{output_ply.stem}_{arch}{output_ply.suffix}")))
    return written
