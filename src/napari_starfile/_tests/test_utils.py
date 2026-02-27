import starfile
from pathlib import Path
import numpy as np
from napari_starfile.utils import euler2vec, vec2euler

def test_euler2vec():
    star = starfile.read(Path(__file__).parent.parent / "data" / "example_particles.star")
    # Check dataframe input
    vecs = euler2vec(star)
    eulers = vec2euler(vecs)
    assert np.allclose(eulers[:, (1, 2)], star[["rlnAngleTilt", "rlnAnglePsi"]].to_numpy())
    # Check array input
    vecs = euler2vec(star[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy())
    eulers = vec2euler(vecs)
    assert np.allclose(eulers[:, (1, 2)], star[["rlnAngleTilt", "rlnAnglePsi"]].to_numpy())