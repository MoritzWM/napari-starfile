"""
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
import warnings
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation


def napari_get_reader(path: Union[str, List[str]]):
    if isinstance(path, str):
        path = [path]
    if not all(p.endswith(".star") for p in path):
        return None
    return read_paths


def read_paths(paths: Union[str, List[str]]) -> List:
    paths = [paths] if isinstance(paths, str) else paths
    layers = []
    for path  in paths:
        layers += read_star(starfile.read(Path(path)))  # type: ignore
    return layers


def read_star(star: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> List:
    if isinstance(star, pd.DataFrame):
        star = {"particles": star}
    if "particles" in star:
        particles = star["particles"]
    elif "" in star:
        particles = star[""]
    else:
        raise ValueError("No particles in star file")
    layers = []
    coords = particles[[f"rlnCoordinate{zyx}" for zyx in "ZYX"]].to_numpy().astype(float)
    shift_columns = [f"rlnOrigin{zyx}Angst" for zyx in "ZYX"]
    if all(col in particles.columns for col in shift_columns):
        warnings.warn("rlnOriginX/Y/ZAngst is not supported yet, ignoring")
        # shifts = particles[shift_columns].to_numpy().astype(float)
        # apix = particles["rlnPixelSize"].to_numpy().astype(float)
        # coords -= shifts/apix[:, None]
    rotations = Rotation.from_euler("ZYZ", particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(), degrees=True).inv()
    layers.append((coords, {"properties": particles}, "points"))
    # Axis vectors
    vecs_z = np.empty((len(coords), 2, 3), dtype=float)
    vecs_z[:, 0] = coords
    vecs_z[:, 1] = rotations.apply([0, 0, 1])[:, ::-1]
    layers.append((vecs_z, {"name": "Z vectors", "edge_color": "blue"}, "vectors"))
    vecs_y = np.empty((len(coords), 2, 3), dtype=float)
    vecs_y[:, 0] = coords
    vecs_y[:, 1] = rotations.apply([0, 1, 0])[:, ::-1]
    layers.append((vecs_y, {"name": "Y vectors", "edge_color": "yellow"}, "vectors"))
    vecs_x = np.empty((len(coords), 2, 3), dtype=float)
    vecs_x[:, 0] = coords
    vecs_x[:, 1] = rotations.apply([1, 0, 0])[:, ::-1]
    layers.append((vecs_x, {"name": "X vectors", "edge_color": "red"}, "vectors"))
    return layers
