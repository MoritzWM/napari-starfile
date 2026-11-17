from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import Container, PushButton, create_widget
from napari.utils.notifications import show_warning
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    import napari

_INDEX_COLUMN = "vector_index"


def _estimate_spacing(points: np.ndarray) -> float:
    """Mean distance from each point to its nearest neighbour."""
    if len(points) < 2:
        return 0.0
    return float(cKDTree(points).query(points, k=2)[0][:, 1].mean())


class EditVectorsWidget(Container):
    """Select and delete vectors through a Points layer.

    napari's Vectors layer cannot be selected from. This widget creates a
    "handles" Points layer with one point at the start of every vector. Select
    handles with napari's Points tools (click, shift-click, box select), then
    delete the corresponding vectors. Deleting handles directly (Backspace)
    deletes their vectors too.
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        self._handles = None
        self._syncing = False

        self._vectors_widget = create_widget(
            annotation="napari.layers.Vectors", label="vectors_layer"
        )
        self._create_button = PushButton(text="Create handles layer")
        self._delete_button = PushButton(text="Delete selected vectors")
        self.extend(
            [self._vectors_widget, self._create_button, self._delete_button]
        )
        self._create_button.changed.connect(self.create_handles)
        self._delete_button.changed.connect(self.delete_selected)

    @property
    def _vectors(self):
        return self._vectors_widget.value

    def _live_handles(self):
        """The handles layer if it still belongs to the viewer and matches."""
        handles = self._handles
        if handles is None or handles not in self._viewer.layers:
            return None
        if self._vectors is None or len(handles.data) != len(self._vectors.data):
            return None
        return handles

    def create_handles(self):
        vectors = self._vectors
        if vectors is None or len(vectors.data) == 0:
            show_warning("Select a vectors layer with at least one vector.")
            return
        if self._handles is not None and self._handles in self._viewer.layers:
            self._disconnect_handles()
            self._viewer.layers.remove(self._handles)
        spacing = _estimate_spacing(vectors.data[:, 0, :])
        handles = self._viewer.add_points(
            vectors.data[:, 0, :],
            features=pd.DataFrame({_INDEX_COLUMN: np.arange(len(vectors.data))}),
            size=spacing * 0.5 if spacing > 0 else 5,
            scale=vectors.scale,
            translate=vectors.translate,
            name=f"{vectors.name} handles",
        )
        handles.events.data.connect(self._on_handles_data)
        self._handles = handles
        self._viewer.layers.selection.active = handles

    def _disconnect_handles(self):
        try:
            self._handles.events.data.disconnect(self._on_handles_data)
        except (TypeError, ValueError):
            pass

    def delete_selected(self):
        handles = self._live_handles()
        if handles is None:
            show_warning("Create a handles layer first.")
            return
        selected = sorted(handles.selected_data)
        if not selected:
            show_warning("Select some handles first.")
            return
        keep = np.ones(len(self._vectors.data), dtype=bool)
        keep[selected] = False
        self._remove_vectors(keep)
        self.create_handles()

    def _remove_vectors(self, keep: np.ndarray):
        vectors = self._vectors
        features = vectors.features[keep].reset_index(drop=True)
        vectors.data = vectors.data[keep]
        vectors.features = features

    def _on_handles_data(self, event):
        """Handles deleted with Backspace: delete their vectors too."""
        if self._syncing:
            return
        handles = self._handles
        vectors = self._vectors
        if handles is None or vectors is None:
            return
        n_vectors = len(vectors.data)
        if len(handles.data) >= n_vectors:
            return
        if len(handles.features) != len(handles.data):
            return
        surviving = handles.features[_INDEX_COLUMN].to_numpy().astype(int)
        if len(surviving) and surviving.max() >= n_vectors:
            return
        keep = np.zeros(n_vectors, dtype=bool)
        keep[surviving] = True
        self._syncing = True
        try:
            self._remove_vectors(keep)
            # Re-number handles to match the shortened vectors layer
            handles.features = pd.DataFrame(
                {_INDEX_COLUMN: np.arange(len(surviving))}
            )
        finally:
            self._syncing = False
