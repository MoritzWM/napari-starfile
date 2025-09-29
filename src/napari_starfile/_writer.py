"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, List

import numpy as np
import pandas as pd
import starfile

from napari_starfile.utils import vec2euler

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def _layers_to_df(
        data: List["FullLayerData"], layer_column_name: str
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
    return pd.concat(dfs)

def write_star_relion3(path: str, data: List["FullLayerData"]) -> List[str]:
    if not path.endswith(".star"):
        path += ".star"
    particles = _layers_to_df(data, "rlnMicrographName")
    starfile.write(particles, Path(path), overwrite=True)
    return [path]