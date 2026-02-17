import warnings
from pathlib import Path

import pandas as pd
import starfile


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
    if "particles" in star and "optics" in star:
        try:
            particles = pd.merge(
                star["particles"],
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
    elif "" in star:
        particles = star[""]
    else:
        raise ValueError("No particles in star file")
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
    return [(coords, {"properties": particles}, "points")]


# rotations = Rotation.from_euler(
# "ZYZ",
# particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(),
# degrees=True,
# ).inv()
