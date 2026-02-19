from __future__ import annotations

from pathlib import Path

from napari_starfile._reader import read_stars


def make_sample_data():
    return read_stars(
        Path(__file__).parent / "data" / "example_particles.star"
    )
