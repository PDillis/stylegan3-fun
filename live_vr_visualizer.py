# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import os
import cv2
import torch
from network_features import VGG16FeaturesNVIDIA

import multiprocessing
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz import renderer
from viz import pickle_widget
from viz import latent_widget
from viz import stylemix_widget
from viz import trunc_noise_widget
from viz import class_widget
from viz import performance_widget
from viz import capture_widget
from viz import layer_widget
from viz import equivariance_widget
from viz import visualreactive_mode_widget
from viz import camera_widget

#----------------------------------------------------------------------------


def setup_vgg16(device: str):
    """Set up VGG16 feature extractor."""
    print('Loading VGG16 and its features...')
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)


class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='GAN Visualizer', window_width=3840, window_height=2160)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0

        # Widgets.
        self.pickle_widget      = pickle_widget.PickleWidget(self)
        self.latent_widget      = latent_widget.LatentWidget(self)
        self.stylemix_widget    = stylemix_widget.StyleMixingWidget(self)
        self.trunc_noise_widget = trunc_noise_widget.TruncationNoiseWidget(self)
        self.class_widget       = class_widget.ClassWidget(self)
        self.perf_widget        = performance_widget.PerformanceWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)
        self.layer_widget       = layer_widget.LayerWidget(self)
        self.eq_widget          = equivariance_widget.EquivarianceWidget(self)

        # New widget for mode selection
        self.mode_widget        = visualreactive_mode_widget.ModeSelectionWidget(self)
        self.mode               = 'v0'  # Default mode

        self.vgg16_features = setup_vgg16('cuda')
        self.static_w = None
        self.mode_args = dnnlib.EasyDict()  # New dictionary for mode-specific arguments

        # Add the camera widget
        self.camera_widget = camera_widget.CameraWidget(self)
        self.camera_height_ratio = 0.3  # Camera takes up 30% of the window height; adjust as needed

        if capture_dir is not None:
            self.capture_widget.path = capture_dir

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.

        self._G = None
        self._pkl = None
        self.current_w = None

    @property
    def G(self):
        if self._G is None and self._pkl is not None:
            try:
                self._G = self._async_renderer.get_network(self._pkl, 'G_ema')
            except Exception as e:
                print(f"Error loading generator: {e}")
                return None
        return self._G

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None
        if self.camera_widget.cap is not None:
            self.camera_widget.cap.release()

    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)
        self._pkl = pkl
        self._G = None  # Reset G so it will be reloaded when accessed

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def process_v0(self, frame):
        if self.G is None:
            print("Error: No generator (G) available. Please load a valid pickle file.")
            return None

        if self.current_w is None:
            print('Error: No latent available. Please generate an image first.')
            return None

        # Convert frame to tensor and preprocess
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to('cuda')
        frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]

        # Extract features
        with torch.no_grad():
            features = self.vgg16_features.get_layers_features(frame_tensor, layers=[self.mode_args.v0_layer])[0]

        # Process features (example: use as latent code)
        fake_z = features.view(1, -1).mean(1, keepdim=True)
        fake_w = self.G.mapping(fake_z, None)

        # Use the current latent as static_w
        static_w = self.current_w.clone()

        fake_w[:, 4:] = static_w[:, 4:]  # Keep the first 4 layers

        # Generate image
        with torch.no_grad():
            img = self.G.synthesis(fake_w)

        # Convert to numpy array
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()

        return img

    def draw_frame(self):
        self.begin_frame()
        self.args = dnnlib.EasyDict()
        self.mode_args = dnnlib.EasyDict()  # Reset mode-specific arguments
        self.pane_w = self.font_size * 45
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)
        self.mode = 'v0'  # Default mode

        # Calculate dimensions for result area and camera area
        result_area_width = self.content_width - self.pane_w
        result_area_height = self.content_height
        camera_height = int(result_area_height * self.camera_height_ratio) if self.camera_widget.camera_enabled else 0
        main_result_height = result_area_height - camera_height

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=result_area_width, height=main_result_height)
        if dragging:
            self.latent_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.latent_widget(expanded)
        self.stylemix_widget(expanded)
        self.trunc_noise_widget(expanded)
        self.class_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & capture', default=True)
        self.perf_widget(expanded)
        self.capture_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Layers & channels', default=True)
        self.layer_widget(expanded)
        with imgui_utils.grayed_out(not self.result.get('has_input_transform', False)):
            expanded, _visible = imgui_utils.collapsing_header('Affine Transformations', default=True)
            self.eq_widget(expanded)

        # New mode selection widget
        expanded, _visible = imgui_utils.collapsing_header('Interaction Mode', default=True)
        self.mode_widget(expanded)

        # Add camera widget to the control pane
        expanded, _visible = imgui_utils.collapsing_header('Camera', default=True)
        self.camera_widget(expanded)

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            render_args = {k: v for k, v in self.args.items() if k != 'mode'}
            self._async_renderer.set_args(**render_args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result
                # Store the current latent
                self.current_w = result.get('w', None)

        # Calculate positions for synthesized image and error messages
        synth_pos = np.array([self.pane_w + result_area_width / 2, main_result_height / 2])

        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)

            synth_zoom = min(result_area_width / self._tex_obj.width, main_result_height / self._tex_obj.height)
            synth_zoom = np.floor(synth_zoom) if synth_zoom >= 1 else synth_zoom
            self._tex_obj.draw(pos=synth_pos, zoom=synth_zoom, align=0.5, rint=True)

            if hasattr(self, 'mode_widget'):
                mode_text = f"Mode: {self.mode_widget.current_mode}"
                cv2.putText(self._tex_img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mode-specific processing
        if self.camera_widget.camera_enabled and 'image' in self.result and self.G is not None:
            camera_frame = self.camera_widget.get_frame()
            if camera_frame is not None:
                if self.mode == 'v0':
                    processed_image = self.process_v0(camera_frame)
                    if processed_image is not None:
                        # Update self._tex_img with processed_image
                        self._tex_img = processed_image
                        self._tex_obj.update(self._tex_img)
                        self.result['image'] = processed_image
                    else:
                        print("Failed to process image in v0 mode.")

        # Update and draw camera feed if enabled
        if self.camera_widget.camera_enabled:
            self.camera_widget.update_texture(gl_utils)
            if self.camera_widget.camera_texture is not None:
                camera_width = result_area_width
                camera_height = min(int(camera_width * 0.5), camera_height)  # Maintain 2:1 aspect ratio
                camera_pos = np.array([self.pane_w + result_area_width / 2, self.content_height - camera_height / 2])
                camera_zoom = min(camera_width / self.camera_widget.camera_texture.width,
                                  camera_height / self.camera_widget.camera_texture.height)
                self.camera_widget.draw(camera_pos, zoom=camera_zoom)


        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=result_area_width,
                                         max_height=main_result_height, outline=2)
            tex.draw(pos=synth_pos, align=0.5, rint=True, color=1)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp

#----------------------------------------------------------------------------

@click.command()
@click.argument('pkls', metavar='PATH', nargs=-1)
@click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
def main(
    pkls,
    capture_dir,
    browse_dir
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    viz = Visualizer(capture_dir=capture_dir)

    if browse_dir is not None:
        viz.pickle_widget.search_dirs = [browse_dir]

    # List pickles.
    if len(pkls) > 0:
        for pkl in pkls:
            viz.add_recent_pickle(pkl)
        viz.load_pickle(pkls[0])
    else:
        pretrained = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl'
        ]

        # Populate recent pickles list with pretrained model URLs.
        for url in pretrained:
            viz.add_recent_pickle(url)

    # Run.
    while not viz.should_close():
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
