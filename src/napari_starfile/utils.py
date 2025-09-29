import numpy as np

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

def euler2vec(angles: np.ndarray) -> np.ndarray:
    """
    Angles are [rot, tilt, psi] in radians.
    Vectors are [Z, Y, X].
    Copy-pasted from https://github.com/3dem/relion/blob/d476e6f6a4f1f37627c06ace5227fc374c0c2b05/src/euler.cpp#L94
    """
    angles = angles.astype(float)
    angles = angles.reshape((-1, 3))
    vec = np.empty_like(angles, dtype=float)
    ca = np.cos(angles[:, 0])
    cb = np.cos(angles[:, 1])
    sa = np.sin(angles[:, 0])
    sb = np.sin(angles[:, 1])
    sc = sb * ca
    ss = sb * sa
    vec[:, 2] = sc
    vec[:, 1] = ss
    vec[:, 0] = cb
    return vec


def vec2euler(vec: np.ndarray) -> np.ndarray:
    """
    Angles are [rot, tilt, psi] in radians.
    Vectors are [Z, Y, X].
    Copy-pasted from https://github.com/3dem/relion/blob/d476e6f6a4f1f37627c06ace5227fc374c0c2b05/src/euler.cpp#L119C1-L143C2
    """
    vec = vec.astype(float).reshape((-1, 3))
    out = np.zeros_like(vec)
    # Normalize vec
    vec /= np.linalg.norm(vec, axis=1)
    # Tilt (b) should be [0, +180] degrees. Rot (a) should be [-180, +180] degrees
    out[:, 0] = np.atan2(vec[:, 1], vec[:, 2])
    out[:, 1] = np.acos(vec[:, 0])

    # If tilt (b) = 0 or 180 degrees, sin(b) = 0, rot (a) cannot be calculated from the direction
    # if np.abs(beta) < 0.001 or np.abs(beta - 180.) < 0.001:
        # alpha = 0.
    return out
