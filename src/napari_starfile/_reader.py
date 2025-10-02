"""
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from pathlib import Path
from typing import Dict, Union, List, cast

import numpy as np
import pandas as pd
import starfile

from napari_starfile import utils


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
        shifts = particles[shift_columns].to_numpy().astype(float)
        apix = particles["rlnPixelSize"].to_numpy().astype(float)
        coords -= shifts/apix[:, None]
    angles = np.deg2rad(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy())
    rot_matrices = np.linalg.inv(utils.euler2matrix(angles, False))
    layers.append((coords, {"properties": particles}, "points"))
    # Axis vectors
    vecs_z = np.empty((len(coords), 2, 3), dtype=float)
    vecs_z[:, 0] = coords
    vecs_z[:, 1] =  rot_matrices @ np.array([1, 0, 0])
    layers.append((vecs_z, {"name": "Z vectors", "edge_color": "blue"}, "vectors"))
    vecs_y = np.empty((len(coords), 2, 3), dtype=float)
    vecs_y[:, 0] = coords
    vecs_y[:, 1] = rot_matrices @ np.array([0, 1, 0])
    layers.append((vecs_y, {"name": "Y vectors", "edge_color": "yellow"}, "vectors"))
    vecs_x = np.empty((len(coords), 2, 3), dtype=float)
    vecs_x[:, 0] = coords
    vecs_x[:, 1] = rot_matrices @ np.array([0, 0, 1])
    layers.append((vecs_x, {"name": "X vectors", "edge_color": "red"}, "vectors"))
    return layers
