from warnings import warn

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.spatial.transform import Rotation


def particles2vecs(
    particles: pd.DataFrame, optics: pd.DataFrame | None
) -> np.ndarray:
    """Converts a particles DataFrame to an (N, 2, 3) array of coords and vectors.
    Vectors are unit vectors in the direction of the Z axis after rotation, axis order is ZYX."""
    if not all(
        col in particles.columns
        for col in [f"rlnCoordinate{xyz}" for xyz in "XYZ"]
    ):
        raise ValueError(
            "Particles DataFrame must contain rlnCoordinateX/Y/Z columns"
        )
    euler_columns = [f"rlnAngle{angle}" for angle in ["Rot", "Tilt", "Psi"]]
    has_eulers = all(col in particles.columns for col in euler_columns)
    if not has_eulers:
        warn(
            "Particles DataFrame does not contain rlnAngleRot/Tilt/Psi columns"
        )
    shift_columns = [f"rlnOrigin{xyz}Angst" for xyz in "XYZ"]
    has_shifts = all(col in particles.columns for col in shift_columns)
    if not has_shifts:
        warn(
            "Particles DataFrame does not contain rlnOriginX/Y/ZAngst columns"
        )
    if optics is not None:
        particles = pd.merge(
            particles,
            optics,
            how="left",
            left_on="rlnOpticsGroup",
            right_on="rlnOpticsGroup",
            validate="many_to_one",
        )
    angpix_column: str | None = None
    for column in [
        "rlnPixelSize",
        "rlnDetectorPixelSize",
        "rlnImagePixelSize",
    ]:
        if column in particles.columns:
            angpix_column = column
            break
    if angpix_column is None:
        warn(
            "No pixel size found in particles or optics, shifts will be ignored"
        )
    coords = (
        particles[[f"rlnCoordinate{xyz}" for xyz in "XYZ"]]
        .to_numpy()
        .astype(float)
    )
    if has_shifts:
        shifts = particles[shift_columns].to_numpy().astype(float)
    else:
        shifts = np.zeros_like(coords)
    if angpix_column is not None:
        shifts /= particles[angpix_column].to_numpy().astype(float)[:, None]
    coords -= shifts

    # Only in the end are the coordinates converted from XYZ to ZYX
    vecs = np.empty((len(coords), 2, 3), dtype=float)
    vecs[:, 0, :] = coords[:, ::-1]
    if has_eulers:
        vecs[:, 1, :] = euler2vec(particles, vector=[0, 0, 1])[:, ::-1]
    else:
        vecs[:, 1, :] = [1, 0, 0]
    return vecs


def vecs2particles(vecs: np.ndarray) -> pd.DataFrame:
    eulers = vec2euler(vecs[:, 1])
    df = pd.DataFrame(
        data={
            "rlnCoordinateX": vecs[:, 0, 2],
            "rlnCoordinateY": vecs[:, 0, 1],
            "rlnCoordinateZ": vecs[:, 0, 0],
            "rlnAngleRot": eulers[:, 0],
            "rlnAngleTilt": eulers[:, 1],
            "rlnAnglePsi": eulers[:, 2],
        }
    )
    return df


def euler2vec(
    euler: np.ndarray | pd.DataFrame, vector: npt.ArrayLike | None = None
) -> np.ndarray:
    """Turns a set of euler angles ((N, 3) array in rot, tilt, psi order or dataframe with rlnAngleRot/Tilt/Psi columns)
    into a unit vector in the direction of the Z axis after rotation."""
    if isinstance(euler, pd.DataFrame):
        euler = euler[
            ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
        ].to_numpy()
    if vector is None:
        # By default, use vector pointing in Z direction
        vector = [0, 0, 1]
    rotations = Rotation.from_euler(
        "ZYZ",
        euler,
        degrees=True,
    ).inv()
    return rotations.apply(vector)


def vec2euler(vecs: np.ndarray) -> np.ndarray:
    """Turns a (N, 3) array of unit vectors into an array of euler angles in rot, tilt, psi order.
    Angles are in degrees."""
    eulers = np.empty_like(vecs, dtype=float)
    for i, vec in enumerate(vecs):
        rot, _ = Rotation.align_vectors(vec[::-1], [[0, 0, 1]])
        eulers[i] = rot.inv().as_euler("ZYZ", degrees=True)
    return eulers
