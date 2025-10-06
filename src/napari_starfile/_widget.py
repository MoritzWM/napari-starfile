"""
- Widget specification: https://napari.org/stable/plugins/building_a_plugin/guides.html#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import Container, create_widget, RadioButtons, ComboBox, Select, RangeSlider
import numpy as np

if TYPE_CHECKING:
    import napari

# if we want even more control over our widget, we can use
# magicgui `Container`
class SubsetSelectorWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self.cb_points_layer = create_widget(
            label="Points", annotation="napari.layers.Points"
        )
        self.rb_select_or_view = RadioButtons(choices=["Select", "Filter"])

        # connect your own callbacks
        self.cb_points_layer.changed.connect(self.on_cb_points_layer_changed)
        self.rb_select_or_view.changed.connect(self.on_rb_select_or_view_changed)
        self.cb_filter_property = ComboBox(choices=[])
        self.cb_filter_property.changed.connect(self.on_cb_filter_property_changed)
        self.cb_discrete_filter = Select(choices=[], allow_multiple=True)
        self.cb_discrete_filter.changed.connect(self.on_cb_discrete_filter_changed)
        self.rs_float_filter = RangeSlider()
        self.rs_float_filter.changed.connect(self.on_rs_float_filter_changed)

        # append into/extend the container with your widgets
        self.extend(
            [
                self.cb_points_layer,
                self.rb_select_or_view,
                self.cb_filter_property,
                self.cb_discrete_filter,
                self.rs_float_filter,
            ]
        )
        self.native_parent_changed.connect(self.on_cb_points_layer_changed)

    def on_cb_points_layer_changed(self):
        layer = self.cb_points_layer.value
        self.cb_filter_property.choices = list(layer.properties.keys())

    def on_rb_select_or_view_changed(self):
        mode = self.rb_select_or_view.value
        print(mode)

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
            self.rs_float_filter.value = (self.rs_float_filter.min, self.rs_float_filter.max)


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
        points_layer.shown = (points_layer.properties[filter_column] >= self.rs_float_filter.value[0]) & (points_layer.properties[filter_column] <= self.rs_float_filter.value[1])
