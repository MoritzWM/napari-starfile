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
    return read_stars


def read_stars(path: Union[str, List[str]]):
    path = [path] if isinstance(path, str) else path
    layers = []
    for p  in path:
        star = cast(
            Dict[str, pd.DataFrame],
            starfile.read(Path(p), always_dict=True)
        )
        if "particles" in star:
            particles = star["particles"]
        elif "" in star:
            particles = star[""]
        else:
            raise ValueError(f"No particles in star file {path}")
        assert isinstance(particles, pd.DataFrame)
        coords = particles[[f"rlnCoordinate{zyx}" for zyx in "ZYX"]].to_numpy()
        shifts = particles[[f"rlnOrigin{zyx}Angst" for zyx in "ZYX"]].to_numpy()
        apix = particles["rlnPixelSize"].to_numpy()
        shifts_px = shifts/apix[:, None]
        angles = np.deg2rad(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy())
        rot_matrices = np.linalg.inv(utils.euler2matrix(angles, False))
        layers.append((coords - shifts_px, {"properties": particles}, "points"))
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