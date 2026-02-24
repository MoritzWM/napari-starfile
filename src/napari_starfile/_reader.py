import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation


def napari_get_reader(path: str | list[str]):
    if isinstance(path, str):
        path = [path]
    if not all(p.endswith(".star") for p in path):
        return None
    return read_stars


def read_stars(paths: str | list[str] | Path | list[Path]) -> list:
    paths = [paths] if isinstance(paths, str) else paths
    paths = [Path(p) for p in paths]
    layers = []
    for path in paths:
        star = starfile.read(path, always_dict=True)
        if "particles" in star:
            particles = star["particles"]
        elif "" in star:
            particles = star[""]
        else:
            raise ValueError("No particles in star file")
        assert isinstance(particles, pd.DataFrame)
        coords = (
            particles[[f"rlnCoordinate{zyx}" for zyx in "ZYX"]]
            .to_numpy()
            .astype(float)
        )
        shift_columns = [f"rlnOrigin{zyx}Angst" for zyx in "ZYX"]
        if all(col in particles.columns for col in shift_columns):
            warnings.warn(
                "rlnOriginX/Y/ZAngst is not supported yet, ignoring",
                stacklevel=2,
            )
            # shifts = particles[shift_columns].to_numpy().astype(float)
            # apix = particles["rlnPixelSize"].to_numpy().astype(float)
            # coords -= shifts/apix[:, None]
            #
        rotations = Rotation.from_euler(
            "ZYZ",
            particles[
                ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
            ].to_numpy(),
            degrees=True,
        ).inv()
        vecs = np.empty((len(coords), 2, 3), dtype=float)
        vecs[:, 0] = coords
        vecs[:, 1] = rotations.apply([0, 0, 1])[:, ::-1]
        extra_kwargs = {"name": path.stem, "edge_color": "blue", "features": particles}
        if "optics" in star:
            extra_kwargs["metadata"] = {"optics": star["optics"]}
        layers.append(
            (vecs, extra_kwargs, "vectors")
        )
    return layers
