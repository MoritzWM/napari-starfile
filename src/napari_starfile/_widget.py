"""
- Widget specification: https://napari.org/stable/plugins/building_a_plugin/guides.html#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui import magic_factory
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    RangeSlider,
    Select,
    create_widget,
)

from napari_starfile import utils

if TYPE_CHECKING:
    import napari


class FilterWidget(Container):
    def __init__(self, parent: "SubsetSelectorWidget"):
        super().__init__()
        self.parent = parent
        self._points_layer = None
        # use create_widget to generate widgets from type annotations
        self.cb_filter_property = ComboBox(choices=[], nullable=True)
        self.cb_filter_property.changed.connect(
            self.on_cb_filter_property_changed
        )
        self.cb_discrete_filter = Select(choices=[], allow_multiple=True)
        self.cb_discrete_filter.changed.connect(self.on_filter_changed)
        self.rs_float_filter = RangeSlider()
        self.rs_float_filter.changed.connect(self.on_filter_changed)
        self.cb_discrete_filter.visible = False
        self.rs_float_filter.visible = False

        self.extend(
            [
                self.cb_filter_property,
                self.cb_discrete_filter,
                self.rs_float_filter,
            ]
        )

    @property
    def points_layer(self) -> "napari.layers.Points | None":  # pyright: ignore[reportAttributeAccessIssue]
        return self._points_layer

    @points_layer.setter
    def points_layer(self, layer: "napari.layers.Points | None"):  # pyright: ignore[reportAttributeAccessIssue]
        self._points_layer = layer
        if layer is None:
            self.cb_filter_property.choices = []
            self.cb_discrete_filter.visible = False
            self.rs_float_filter.visible = False
        else:
            self.cb_filter_property.choices = list(layer.properties.keys())

    def on_filter_changed(self):
        if self.parent is not None:
            self.parent.update_mask()

    def get_mask(self) -> np.ndarray | None:
        if self.points_layer is None:
            return None
        filter_column = self.cb_filter_property.value
        if filter_column is None:
            return np.ones(len(self.points_layer.data), dtype=bool)
        if self.cb_discrete_filter.visible:
            mask = np.zeros(len(self.points_layer.data), dtype=bool)
            for val in self.cb_discrete_filter.value:
                mask |= self.points_layer.properties[filter_column] == val
            return mask
        if self.rs_float_filter.visible:
            return (
                self.points_layer.properties[filter_column]
                >= self.rs_float_filter.value[0]
            ) & (
                self.points_layer.properties[filter_column]
                <= self.rs_float_filter.value[1]
            )
        raise ValueError("No filter is visible")

    def on_cb_filter_property_changed(self):
        filter_column = self.cb_filter_property.value
        if self.points_layer is None or filter_column is None:
            return
        values = self.points_layer.properties[filter_column]
        if values.dtype in (int, "O"):
            self.cb_discrete_filter.visible = True
            self.rs_float_filter.visible = False
            self.cb_discrete_filter.choices = sorted(np.unique(values))
        elif values.dtype == float:
            self.cb_discrete_filter.visible = False
            self.rs_float_filter.visible = True
            self.rs_float_filter.min = float(np.min(values))
            self.rs_float_filter.max = float(np.max(values))
            self.rs_float_filter.value = (
                self.rs_float_filter.min,
                self.rs_float_filter.max,
            )

def _first_str_feature(features: "pd.DataFrame") -> str:
    for column in features.columns:
        if pd.api.types.is_string_dtype(features[column].dtype):
            return column
    return features.columns[0]


def _split_widget_init(widget):
    @widget.layer.changed.connect
    def on_layer_changed(layer: "napari.layers.Vectors | None"):  # pyright: ignore[reportAttributeAccessIssue]
        if layer is None or layer.features is None:
            widget.column.choices = []
            return
        widget.column.choices = lambda _: list(layer.features.columns)
        widget.column.value = _first_str_feature(layer.features)
    on_layer_changed(widget.layer.value)

    @widget.column.changed.connect
    def on_column_changed(column: str):
        layer = widget.layer.value
        if layer is None or layer.features is None:
            widget.call_button.text = "Split"
            return
        n_unique_values = layer.features[column].nunique()
        widget.call_button.text = f"Split into {n_unique_values} layers"
    on_column_changed(widget.column.value)

@magic_factory(
    widget_init=_split_widget_init,
    column={"widget_type": "ComboBox"},
    call_button="Split"
)
def split_widget(
    layer: "napari.layers.Vectors", column: str # type: ignore
) -> list["napari.types.LayerDataTuple"]:  # pyright: ignore[reportAttributeAccessIssue]
    if column == "":
        return []
    optics = layer.metadata.get("optics", None)
    layers = []
    for value, table in layer.features.groupby(column):
        vecs = utils.particles2vecs(table, optics)
        extra_kwargs = {"name": value, "edge_color": "blue", "features": table}
        if optics is not None:
            extra_kwargs["metadata"] = {"optics": optics}
        layers.append((vecs, extra_kwargs, "vectors"))
    return layers


class SubsetSelectorWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):  # pyright: ignore[reportAttributeAccessIssue]
        super().__init__()
        self._viewer = viewer
        self.cb_points_layer = create_widget(
            label="Points", annotation="napari.layers.Points"
        )
        self.cb_points_layer.changed.connect(self.on_cb_points_layer_changed)
        self.b_add_filter = PushButton(text="Add filter")
        self.b_pop_filter = PushButton(text="Remove filter")
        self.b_add_filter.clicked.connect(self.on_b_add_filter_clicked)
        self.b_pop_filter.clicked.connect(self.on_b_pop_filter_clicked)
        self.filter_widgets: list[FilterWidget] = []

        # append into/extend the container with your widgets
        self.extend(
            [
                self.cb_points_layer,
                self.b_add_filter,
                self.b_pop_filter,
            ]
        )
        self.on_b_add_filter_clicked()
        self.native_parent_changed.connect(self.on_cb_points_layer_changed)

    def update_mask(self):
        points_layer = self.cb_points_layer.value
        if points_layer is None:
            return
        mask = np.ones(len(points_layer.data), dtype=bool)
        for widget in self.filter_widgets:
            widget_mask = widget.get_mask()
            if widget_mask is not None:
                mask &= widget_mask
        points_layer.shown = mask

    def on_b_add_filter_clicked(self):
        widget = FilterWidget(self)
        widget.points_layer = self.cb_points_layer.value
        self.filter_widgets.append(widget)
        self.extend([widget])
        self.b_pop_filter.enabled = True

    def on_b_pop_filter_clicked(self):
        if len(self.filter_widgets) == 0:
            return
        widget = self.filter_widgets.pop()
        widget.points_layer = None
        self.remove(widget)
        self.b_pop_filter.enabled = len(self.filter_widgets) > 0

    def on_cb_points_layer_changed(self):
        layer = self.cb_points_layer.value
        for widget in self.filter_widgets:
            widget.points_layer = layer

    def on_cb_filter_property_changed(self):
        points_layer = self.cb_points_layer.value
        filter_column = self.cb_filter_property.value
        if points_layer is None or filter_column is None:
            return
        values = points_layer.properties[filter_column]
        if values.dtype == "O":
            self.cb_discrete_filter.visible = True
            self.rs_float_filter.visible = False
            self.cb_discrete_filter.choices = list(np.unique(values))
        elif values.dtype in (int, float):
            self.cb_discrete_filter.visible = False
            self.rs_float_filter.visible = True
            self.rs_float_filter.min = float(np.min(values))
            self.rs_float_filter.max = float(np.max(values))
            self.rs_float_filter.value = (
                self.rs_float_filter.min,
                self.rs_float_filter.max,
            )

    def on_cb_discrete_filter_changed(self):
        points_layer = self.cb_points_layer.value
        filter_column = self.cb_filter_property.value
        if points_layer is None or filter_column is None:
            return
        mask = np.zeros(len(points_layer.data), dtype=bool)
        for val in self.cb_discrete_filter.value:
            mask |= points_layer.properties[filter_column] == val
        points_layer.shown = mask

    def on_rs_float_filter_changed(self):
        points_layer = self.cb_points_layer.value
        filter_column = self.cb_filter_property.value
        if points_layer is None or filter_column is None:
            return
        points_layer.shown = (
            points_layer.properties[filter_column]
            >= self.rs_float_filter.value[0]
        ) & (
            points_layer.properties[filter_column]
            <= self.rs_float_filter.value[1]
        )
