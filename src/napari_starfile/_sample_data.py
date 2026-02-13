from __future__ import annotations
from pathlib import Path
import starfile

from napari_starfile._reader import read_star


def make_sample_data():
    return read_star(starfile.read(Path(__file__).parent / "data" / "example_particles.star"))
