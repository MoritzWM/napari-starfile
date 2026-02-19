"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def write_star_relion3(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    for layer_data, layer_meta, layer_type in data:
        if layer_type != "vectors":
            raise ValueError(f"Unsupported layer type: {layer_type}")
    starfile.write(particles, Path(path), overwrite=True)
    return [path]


def write_star_relion31(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    raise NotImplementedError("write_star_relion31 is not implemented yet")


def write_star_relion5(path: str, data: list["FullLayerData"]) -> list[str]:
    if not path.endswith(".star"):
        path += ".star"
    raise NotImplementedError("write_star_relion5 is not implemented yet")
