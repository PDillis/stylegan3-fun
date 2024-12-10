import imgui
from gui_utils import imgui_utils

# ----------------------------------------------------------------------------


class ModeSelectionWidget:
    def __init__(self, viz):
        self.viz = viz
        self.modes = ['v0', 'v1', 'v2', 'v3', 'v4']
        self.current_mode = 'v0'
        self.v0_layer = 'conv4_1'  # Default layer for v0 mode

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            imgui.text('Mode')
            imgui.same_line(self.viz.label_w)
            changed, value = imgui.combo(
                "##mode", self.modes.index(self.current_mode), self.modes
            )
            if changed:
                self.current_mode = self.modes[value]
                self.viz.args.mode = self.current_mode

            # Add mode-specific parameters
            if self.current_mode == 'v0':
                imgui.text('Layer')
                imgui.same_line(self.viz.label_w)
                _, self.v0_layer = imgui.input_text("##v0_layer", self.v0_layer, 256)
                self.viz.mode_args.v0_layer = self.v0_layer