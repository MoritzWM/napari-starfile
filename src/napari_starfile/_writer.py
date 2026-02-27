"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import pandas as pd
import starfile

from napari_starfile import utils

if TYPE_CHECKING:
    DataType = Any | Sequence[Any]
    FullLayerData = tuple[DataType, dict, str]


def layer2particles(layer_data: "DataType", layer_meta: dict, layer_type: str) -> pd.DataFrame:
    if layer_type != "vectors":
        raise ValueError(f"Unsupported layer type: {layer_type}")
    particles = layer_meta["features"]
    if not isinstance(particles, pd.DataFrame):
        raise ValueError("Layer features must be a DataFrame")
    # Check if the layer has a particles table already
    if len(particles) == 0:
        particles = utils.vecs2particles(layer_data)
    # Can't check coordinates yet because rlnOriginX/Y/ZAngst is not supported yet, but we can check for angles
    if not all(col in particles.columns for col in [f"rlnCoordinate{zyx}" for zyx in "ZYX"]):
        raise ValueError("Particles DataFrame must contain rlnCoordinateX/Y/Z columns")
    if not all(col in particles.columns for col in [f"rlnAngle{angle}" for angle in ["Rot", "Tilt", "Psi"]]):
        raise ValueError("Particles DataFrame must contain rlnAngleRot/Tilt/Psi columns")
    particles["rlnMicrographName"] = layer_meta["name"].replace(" ", "_")
    return particles

def write_star_relion3(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    all_particles: list[pd.DataFrame] = []
    for layer_data, layer_meta, layer_type in data:
        particles = layer2particles(layer_data, layer_meta, layer_type)
        if "optics" in layer_meta["metadata"]:
            try:
                particles = pd.merge(
                    particles,
                    layer_meta["metadata"]["optics"],
                    how="left",
                    left_on="rlnOpticsGroup",
                    right_on="rlnOpticsGroup",
                    validate="many_to_one",
                )
            except pd.errors.MergeError as err:
                raise ValueError(
                    "Error: could not merge particles and optics"
                ) from err
        all_particles.append(particles)
    particles = pd.concat(all_particles, ignore_index=True, join="inner")
    starfile.write(all_particles, Path(path), overwrite=True)
    return [path]


def write_star_relion31(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    all_particles: list[pd.DataFrame] = []
    all_optics: list[pd.DataFrame] | pd.DataFrame = []
    for layer_data, layer_meta, layer_type in data:
        particles = layer2particles(layer_data, layer_meta, layer_type)
        if "optics" in layer_meta["metadata"]:
            all_optics.append(layer_meta["metadata"]["optics"])
        all_particles.append(particles)
    particles = pd.concat(all_particles, ignore_index=True, join="inner")
    if len(all_optics) > 1:
        warnings.warn("Joining optics tables from different layers. Make sure they are all the same!", stacklevel=2)
    optics = pd.concat(all_optics, join="inner").drop_duplicates().reset_index(drop=True)
    starfile.write({"optics": optics, "particles": particles}, Path(path), overwrite=True)
    return [path]

def write_star_relion5(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    raise NotImplementedError("write_star_relion5 is not implemented yet")
