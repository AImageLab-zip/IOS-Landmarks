"""Simple in-memory cache for teeth meshes and their metadata."""
import numpy as np
import trimesh
from pathlib import Path
import json


class TeethCache:
    """In-memory cache for teeth meshes and transformation metadata."""

    def __init__(self):
        self.meshes = {}  # tooth_key -> trimesh object
        self.transforms = {}  # tooth_key -> transform dict
        self.cores = {}  # tooth_key -> (K, 3) undilated vertices
        self.core_meshes = {}  # tooth_key -> undilated trimesh (same frame as meshes[key])
        self.scan_meshes = {}  # scan_name -> trimesh object
    
    def load_mesh(self, teeth_path: Path, tooth_key: str) -> trimesh.Trimesh:
        """Load mesh from cache or disk. Returns trimesh object."""
        if tooth_key in self.meshes:
            return self.meshes[tooth_key]
        
        stl_file = teeth_path / f"{tooth_key}.stl"
        mesh = trimesh.load_mesh(stl_file)
        self.meshes[tooth_key] = mesh
        return mesh
    
    def store_mesh(self, tooth_key: str, mesh: trimesh.Trimesh, transform_data: dict):
        """Store mesh and transformation in cache (skip disk I/O)."""
        self.meshes[tooth_key] = mesh
        self.transforms[tooth_key] = transform_data
    
    def load_transform(self, teeth_path: Path, tooth_key: str) -> dict:
        """Load transformation metadata from cache or disk."""
        if tooth_key in self.transforms:
            return self.transforms[tooth_key]
        
        transform_file = teeth_path / f"{tooth_key}.json"
        with open(transform_file, 'r') as f:
            transform_data = json.load(f)
        self.transforms[tooth_key] = transform_data
        return transform_data
    
    def load_core(self, teeth_path: Path, tooth_key: str):
        """Undilated tooth vertices from cache or disk, None if unavailable.

        These are the vertices carrying the tooth's own segmentation label,
        without the dilation collar of gum and adjacent teeth. Returns None for
        teeth segmented before this was introduced, so callers can degrade
        instead of failing.
        """
        if tooth_key in self.cores:
            return self.cores[tooth_key]

        core_file = teeth_path / f"{tooth_key}.core.npy"
        if not core_file.exists():
            return None
        core = np.load(core_file)
        self.cores[tooth_key] = core
        return core

    def store_core(self, tooth_key: str, core_points):
        """Store undilated tooth vertices in cache (skip disk I/O)."""
        self.cores[tooth_key] = core_points

    def store_core_mesh(self, tooth_key: str, mesh):
        """Store the undilated tooth mesh in cache (skip disk I/O)."""
        self.core_meshes[tooth_key] = mesh

    def clear(self):
        """Clear all cached data."""
        self.meshes.clear()
        self.transforms.clear()
        self.cores.clear()
        self.core_meshes.clear()
        self.scan_meshes.clear()

    def clear_teeth_data(self):
        """Clear only tooth-level cache, keep preloaded scan meshes."""
        self.meshes.clear()
        self.transforms.clear()
        self.cores.clear()
        self.core_meshes.clear()

    def preload_scan_mesh(self, scan_path: Path):
        """Load a full scan mesh once and keep it in memory."""
        scan_name = scan_path.name
        if scan_name not in self.scan_meshes:
            mesh = trimesh.load_mesh(str(scan_path), process=False)
            mesh.merge_vertices()
            self.scan_meshes[scan_name] = mesh

    def get_scan_mesh(self, scan_path: Path):
        """Get a preloaded scan mesh by filename, if available."""
        return self.scan_meshes.get(scan_path.name)
