import warnings
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

def particles2vecs(particles: pd.DataFrame) -> np.ndarray:
    """Converts a particles DataFrame to an (N, 2, 3) array of coords and vectors.
    Vectors are unit vectors in the direction of the Z axis after rotation, axis order is ZYX."""
    if not all(col in particles.columns for col in [f"rlnCoordinate{zyx}" for zyx in "ZYX"]):
        raise ValueError("Particles DataFrame must contain rlnCoordinateX/Y/Z columns")
    if not all(col in particles.columns for col in [f"rlnAngle{angle}" for angle in ["Rot", "Tilt", "Psi"]]):
        raise ValueError("Particles DataFrame must contain rlnAngleRot/Tilt/Psi columns")
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
    
    vecs = np.empty((len(coords), 2, 3), dtype=float)
    vecs[:, 0] = coords
    vecs[:, 1] = euler2vec(particles)
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

def euler2matrix(angles: np.ndarray, homogenous: bool) -> np.ndarray:
    """
    Angles are [rot, tilt, psi] in radians.
    Vectors are [Z, Y, X].
    Copy-pasted from https://github.com/3dem/relion/blob/d476e6f6a4f1f37627c06ace5227fc374c0c2b05/src/euler.cpp#L52
    """
    angles = angles.reshape((-1, 3))
    if homogenous:
        out = np.zeros((len(angles), 4, 4), dtype=float)
        out[:, 3, 3] = 1
    else:
        out = np.zeros((len(angles), 3, 3), dtype=float)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    cc = cos_angles[:, 1] * cos_angles[:, 0]
    cs = cos_angles[:, 1] * sin_angles[:, 0]
    sc = sin_angles[:, 1] * cos_angles[:, 0]
    ss = sin_angles[:, 1] * sin_angles[:, 0]
    out[:, 2, 2] = cos_angles[:, 2] * cc - sin_angles[:, 2] * sin_angles[:, 0]
    out[:, 2, 1] = cos_angles[:, 2] * cs + sin_angles[:, 2] * cos_angles[:, 0]
    out[:, 2, 0] = -cos_angles[:, 2] * sin_angles[:, 1]
    out[:, 1, 2] = -sin_angles[:, 2] * cc - cos_angles[:, 2] * sin_angles[:, 0]
    out[:, 1, 1] = -sin_angles[:, 2] * cs + cos_angles[:, 2] * cos_angles[:, 0]
    out[:, 1, 0] = sin_angles[:, 2] * sin_angles[:, 1]
    out[:, 0, 2] = sc
    out[:, 0, 1] = ss
    out[:, 0, 0] = cos_angles[:, 1]
    return out

def euler2vec(euler: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Turns a set of euler angles ((N, 3) array in rot, tilt, psi order or dataframe with rlnAngleRot/Tilt/Psi columns)
    into a unit vector in the direction of the Z axis after rotation."""
    if isinstance(euler, pd.DataFrame):
        euler = euler[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy()
    rotations = Rotation.from_euler(
        "ZYZ",
        euler,
        degrees=True,
    ).inv()
    return rotations.apply([0, 0, 1])[:, ::-1]

def vec2euler(vecs: np.ndarray) -> np.ndarray:
    """Turns a (N, 3) array of unit vectors into an array of euler angles in rot, tilt, psi order.
    Angles are in degrees."""
    eulers = np.empty_like(vecs, dtype=float)
    for i, vec in enumerate(vecs):
        rot, _ = Rotation.align_vectors(vec[::-1], [[0, 0, 1]])
        eulers[i] = rot.inv().as_euler("ZYZ", degrees=True)
    return eulers