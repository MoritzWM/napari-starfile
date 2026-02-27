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
            particles = star["particles"]
        elif "" in star:
            particles = star[""]
        else:
            raise ValueError("No particles in star file")
        assert isinstance(particles, pd.DataFrame)
        vecs = utils.particles2vecs(particles)
        extra_kwargs = {"name": path.stem, "edge_color": "blue", "features": particles}
        if "optics" in star:
            extra_kwargs["metadata"] = {"optics": star["optics"]}
        layers.append(
            (vecs, extra_kwargs, "vectors")
        )
    return layers
