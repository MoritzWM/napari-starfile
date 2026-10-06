import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation

from napari_starfile import utils


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
            particles: pd.DataFrame = star["particles"] # type: ignore
        elif "" in star:
            particles: pd.DataFrame = star[""] # type: ignore
        else:
            raise ValueError("No particles in star file")
        assert isinstance(particles, pd.DataFrame)
        optics: pd.DataFrame | None = star.get("optics", None) # type: ignore
        vecs = utils.particles2vecs(particles, optics)
        extra_kwargs = {"name": path.stem, "edge_color": "blue", "features": particles}
        if optics is not None:
            extra_kwargs["metadata"] = {"optics": optics}
        layers.append(
            (vecs, extra_kwargs, "vectors")
        )
    return layers
