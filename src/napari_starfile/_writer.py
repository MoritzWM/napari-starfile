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

from napari_starfile.utils import vec2euler

if TYPE_CHECKING:
    DataType = Any | Sequence[Any]
    FullLayerData = tuple[DataType, dict, str]


def _layers_to_df(
    data: list["FullLayerData"], layer_column_name: str
) -> pd.DataFrame:
    """Turn a list of FullLayerData of vectors into a DataFrame with particle
    coordinates.
    """
    dfs = []
    for layer_data, meta, _layer_type in data:
        angles = np.rad2deg(vec2euler(layer_data[:, 1, :], True))
        df = pd.DataFrame(
            data={
                "rlnCoordinateX": layer_data[:, 0, 2],
                "rlnCoordinateY": layer_data[:, 0, 1],
                "rlnCoordinateZ": layer_data[:, 0, 0],
                "rlnAngleRot": angles[:, 0],
                "rlnAngleTilt": angles[:, 1],
                "rlnAnglePsi": angles[:, 2],
                layer_column_name: meta["name"].replace(" ", "_"),
            }
        )
        dfs.append(df)
        coords = (
            particles[[f"rlnCoordinate{zyx}" for zyx in "ZYX"]]
            .to_numpy()
            .astype(float)
        )
        shift_columns = [f"rlnOrigin{zyx}Angst" for zyx in "ZYX"]
        if all(col in particles.columns for col in shift_columns):
            warnings.warn("rlnOriginX/Y/ZAngst is not supported yet, ignoring")
            # shifts = particles[shift_columns].to_numpy().astype(float)
            # apix = particles["rlnPixelSize"].to_numpy().astype(float)
            # coords -= shifts/apix[:, None]
        rotations = Rotation.from_euler(
            "ZYZ",
            particles[
                ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
            ].to_numpy(),
            degrees=True,
        ).inv()
    return pd.concat(dfs)

def _verify_table(df: pd.DataFrame):
        required_columns = [f"rlnCoordinate{zyx}" for zyx in "ZYX"] + [f"rlnAngle{angle}" for angle in ["Rot", "Tilt", "Psi"]]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

def write_star_relion3(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    all_particles: list[pd.DataFrame] = []
    for layer_data, layer_meta, layer_type in data:
        if layer_type != "vectors":
            raise ValueError(f"Unsupported layer type: {layer_type}")
        particles = layer_meta["features"]
        _verify_table(particles)
        particles["rlnMicrographName"] = layer_meta["name"].replace(" ", "_")
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
        if layer_type != "vectors":
            raise ValueError(f"Unsupported layer type: {layer_type}")
        particles = layer_meta["features"]
        _verify_table(particles)
        particles["rlnMicrographName"] = layer_meta["name"].replace(" ", "_")
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
