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
    return read_paths


def read_paths(paths: str | list[str]) -> list:
    paths = [paths] if isinstance(paths, str) else paths
    layers = []
    for path in paths:
        layers += read_star(starfile.read(Path(path)))  # type: ignore
    return layers


def read_star(star: dict[str, pd.DataFrame] | pd.DataFrame) -> list:
    if isinstance(star, pd.DataFrame):
        star = {"particles": star}
    if "particles" in star:
        particles = star["particles"]
    elif "particles" in star:
        particles = star[""]
    else:
        raise ValueError("No particles in star file")
    if "optics" in star:
        try:
            particles = pd.merge(
                particles,
                star["optics"],
                how="left",
                left_on="rlnOpticsGroup",
                right_on="rlnOpticsGroup",
                validate="many_to_one",
            )
        except pd.errors.MergeError as err:
            raise ValueError(
                "Error: could not merge particles and optics"
            ) from err
    coords = (
        particles[[f"rlnCoordinate{zyx}" for zyx in "ZYX"]]
        .to_numpy()
        .astype(float)
    )
    shift_columns = [f"rlnOrigin{zyx}Angst" for zyx in "ZYX"]
    if all(col in particles.columns for col in shift_columns):
        warnings.warn(
            "rlnOriginX/Y/ZAngst is not supported yet, ignoring", stacklevel=2
        )
        # shifts = particles[shift_columns].to_numpy().astype(float)
        # apix = particles["rlnPixelSize"].to_numpy().astype(float)
        # coords -= shifts/apix[:, None]
        #
    rotations = Rotation.from_euler(
        "ZYZ",
        particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(),
        degrees=True,
    ).inv()
    vecs_z = np.empty((len(coords), 2, 3), dtype=float)
    vecs_z[:, 0] = coords
    vecs_z[:, 1] = rotations.apply([0, 0, 1])[:, ::-1]
    return [
        (
            vecs_z,
            {
                "name": "Z vectors",
                "vector_style": "arrow",
                "properties": particles,
            },
            "vectors",
        )
    ]
    # return [(coords, {"properties": particles}, "points")]
