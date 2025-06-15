import os
import re
import json

from typing import List, Tuple, Union, Optional, Type
from collections import OrderedDict
from locale import atof

import click
import numpy as np
import torch

try:
    import dnnlib
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..',))
    import dnnlib
import legacy


# ----------------------------------------------------------------------------


channels_dict = {1: 'L', 3: 'RGB', 4: 'RGBA'}


# ----------------------------------------------------------------------------


available_cfgs = ['stylegan2', 'stylegan2-ext', 'stylegan3-t', 'stylegan3-r']


# ----------------------------------------------------------------------------


def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns in the grid must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


# ----------------------------------------------------------------------------


def parse_fps(fps: Union[str, int]) -> int:
    """Return FPS for the video; at worst, video will be 1 FPS, but no lower.
    Useful if we don't have click, else simply use click.IntRange(min=1)"""
    if isinstance(fps, int):
        return max(fps, 1)
    try:
        fps = int(atof(fps))
        return max(fps, 1)
    except ValueError:
        print(f'Typo in "--fps={fps}", will use default value of 30')
        return 30


def num_range(s: str, remove_repeated: bool = False) -> List[int]:
    """
    Extended helper function from the original (original is contained here).
    Accept a comma separated list of numbers 'a,b,c', a range 'a-c', or a combination
    of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...', and return as a list of ints.
    """
    nums = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for el in s.split(','):
        match = range_re.match(el)
        if match:
            # Sanity check 1: accept ranges 'a-b' or 'b-a', with a<=b
            lower, upper = int(match.group(1)), int(match.group(2))
            if lower <= upper:
                r = list(range(lower, upper + 1))
            else:
                r = list(range(upper, lower + 1))
            # We will extend nums as r is also a list
            nums.extend(r)
        else:
            # It's a single number, so just append it (if it's an int)
            try:
                nums.append(int(atof(el)))
            except ValueError:
                continue  # we ignore bad values
    # Sanity check 2: delete repeating numbers by default, but keep order given by user
    if remove_repeated:
        nums = list(OrderedDict.fromkeys(nums))
    return nums


def float_list(s: str) -> List[float]:
    """
    Helper function for parsing a string of comma-separated floats and returning each float
    """
    str_list = s.split(',')
    nums = []
    float_re = re.compile(r'^(\d+.\d+)$')
    for el in str_list:
        match = float_re.match(el)
        if match:
            nums.append(float(match.group(1)))
        else:
            try:
                nums.append(float(el))
            except ValueError:
                continue  # Ignore bad values

    return nums


def parse_slowdown(slowdown: Union[str, int]) -> int:
    """Function to parse the 'slowdown' parameter by the user. Will approximate to the nearest power of 2."""
    # TODO: slowdown should be any int
    if not isinstance(slowdown, int):
        try:
            slowdown = atof(slowdown)
        except ValueError:
            print(f'Typo in "{slowdown}"; will use default value of 1')
            slowdown = 1
    assert slowdown > 0, '"slowdown" cannot be negative or 0!'
    # Let's approximate slowdown to the closest power of 2 (nothing happens if it's already a power of 2)
    slowdown = 2**int(np.rint(np.log2(slowdown)))
    return max(slowdown, 1)  # Guard against 0.5, 0.25, ... cases


def parse_new_center(s: str) -> Tuple[str, Union[int, Tuple[np.ndarray, Optional[str]]]]:
    """Get a new center for the W latent space (a seed or projected dlatent; to be transformed later)"""
    try:
        new_center = int(s)  # it's a seed
        return s, new_center
    except ValueError:
        new_center = get_latent_from_file(s, return_ext=False)  # it's a projected dlatent
        return s, new_center


def parse_all_projected_dlatents(s: str) -> List[torch.Tensor]:
    """Get all the dlatents (.npy/.npz files) in a given directory"""
    # Get all the files in the directory and subdirectories
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(s)) for f in fn]
    # Filter only the .npy or .npz files
    files = [f for f in files if f.endswith('.npy') or f.endswith('.npz')]
    # Sort them by name, but only according to the last digits in the name (in case there's an error before)
    files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    # Get the full path
    # files = [os.path.join(s, f) for f in files]
    # Get the dlatents
    dlatents = [get_latent_from_file(f, return_ext=False) for f in files]

    return dlatents


def load_network(name: str, network_pkl: Union[str, os.PathLike], cfg: Optional[str], device: torch.device):
    """Load and return the D/G/G_ema from a trained network."""
    # Define the model
    if cfg is not None:
        assert network_pkl in resume_specs[cfg], f'{network_pkl} is not available for config {cfg}! \nAvailable models: {resume_specs[cfg]}'
        network_pkl = resume_specs[cfg][network_pkl]
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = legacy.load_network_pkl(f)[name].eval().requires_grad_(False).to(device)  # type: ignore

    return net


def parse_class(net, class_idx: int, ctx: click.Context) -> Union[int, Type[None]]:
    """Parse the class_idx and return it, if it's allowed by the conditional network"""
    if net.c_dim == 0:
        # Unconditional model
        return None
    # Conditional model, so class must be specified by user
    if class_idx is None:
        ctx.fail('Must specify class label with --class when using a conditional network!')
    if class_idx not in range(net.c_dim):
        ctx.fail(f'Your class label can be at most {net.c_dim - 1}!')
    print(f'Using class {class_idx} (available labels: range({net.c_dim - 1})...)')
    return class_idx


# ----------------------------------------------------------------------------


def save_video_from_images(run_dir: str,
                           image_names: str,
                           video_name: str,
                           fps: int = 30,
                           reverse_video: bool = True,
                           crf: int = 20,
                           pix_fmt: str = 'yuv420p') -> None:
    """ Save a .mp4 video from the images in the run_dir directory; the video can also be saved in reverse """
    print('Saving video...')
    try:
        import ffmpeg
    except ImportError:
        raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')
    # Get the ffmpeg command for the current OS (not tested in MacOS!)
    if os.name == 'nt':
        ffmpeg_command = r'C:\\Ffmpeg\\bin\\ffmpeg.exe'
    else:
        # Get where is the ffmpeg command via `whereis ffmpeg` in the terminal
        ffmpeg_command = os.popen('whereis ffmpeg').read().split(' ')[1:]
        # Remove any ffprobe and ffplay commands
        ffmpeg_command = [c for c in ffmpeg_command if 'ffprobe' not in c and 'ffplay' not in c]
        # If there are more, just select the first one and remove the newline character
        ffmpeg_command = ffmpeg_command[0].replace('\n', '')

    stream = ffmpeg.input(os.path.join(run_dir, image_names), framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, f'{video_name}.mp4'), crf=crf, pix_fmt=pix_fmt)
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, cmd=ffmpeg_command)

    # Save the reversed video apart from the original one, so the user can compare both
    if reverse_video:
        stream = ffmpeg.input(os.path.join(run_dir, f'{video_name}.mp4'))
        stream = stream.video.filter('reverse')
        stream = ffmpeg.output(stream, os.path.join(run_dir, f'{video_name}_reversed.mp4'), crf=crf, pix_fmt=pix_fmt)
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # ibidem


def compress_video(
        original_video: Union[str, os.PathLike],
        original_video_name: Union[str, os.PathLike],
        outdir: Union[str, os.PathLike],
        ctx: click.Context = None) -> None:
    """ Helper function to compress the original_video using ffmpeg-python. moviepy creates huge videos, so use
        ffmpeg to 'compress' it (won't be perfect, 'compression' will depend on the video dimensions). ffmpeg
        can also be used to e.g. resize the video, make a GIF, save all frames in the video to the outdir, etc.
    """
    try:
        import ffmpeg
    except (ModuleNotFoundError, ImportError):
        if ctx is not None:
            ctx.fail('Missing ffmpeg! Install it via "pip install ffmpeg-python"')
        else:
            raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')

    print('Compressing the video...')
    resized_video_name = os.path.join(outdir, f'{original_video_name}-compressed.mp4')
    ffmpeg.input(original_video).output(resized_video_name).run(capture_stdout=True, capture_stderr=True)
    print('Success!')


# ----------------------------------------------------------------------------


def interpolation_checks(
        t: Union[float, np.ndarray],
        v0: np.ndarray,
        v1: np.ndarray) -> Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:
    """Tests for the interpolation functions"""
    # Make sure 0.0<=t<=1.0
    assert np.min(t) >= 0.0 and np.max(t) <= 1.0
    # Guard against v0 and v1 not being NumPy arrays
    if not isinstance(v0, np.ndarray):
        v0 = np.array(v0)
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    # Both should have the same shape in order to interpolate between them
    assert v0.shape == v1.shape, f'Incompatible shapes! v0: {v0.shape}, v1: {v1.shape}'
    return t, v0, v1


def lerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray]) -> np.ndarray:
    """
    Linear interpolation between v0 (starting) and v1 (final) vectors; for optimal results,
    use t as an np.ndarray to return all results at once via broadcasting
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    v2 = (1.0 - t) * v0 + t * v1
    return v2


def slerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        dot_threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical linear interpolation between v0 (starting) and v1 (final) vectors; for optimal
    results, use t as an np.ndarray to return all results at once via broadcasting.

    dot_threshold is the threshold for considering if the two vectors are collinear (not recommended to alter).

    Adapted from the Python code at: https://en.wikipedia.org/wiki/Slerp (at the time, now no longer available).
    Most likely taken from Jonathan Blow's code in C++:
            http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    # Copy vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't always use np.dot, so we use the definition)
    dot = np.sum(v0 * v1)
    # If it's ~1, vectors are ~colineal, so use lerp on the original vectors
    if np.abs(dot) > dot_threshold:
        return lerp(t, v0_copy, v1_copy)
    # Stay within domain of arccos
    dot = np.clip(dot, -1.0, 1.0)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Divide the angle into t steps
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def interpolate(
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        n_steps: int,
        interp_type: str = 'spherical',
        smooth: bool = False) -> np.ndarray:
    """
    Interpolation function between two vectors, v0 and v1. We will either do a 'linear' or 'spherical' interpolation,
    taking n_steps. The steps can be 'smooth'-ed out, so that the transition between vectors isn't too drastic.
    """
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False)
    # TODO: have a dictionary with easing functions that contains my 'smooth' one (might be useful for someone else)
    if smooth:
        # Smooth out the interpolation with a polynomial of order 3 (cubic function f)
        # Constructed f by setting f'(0) = f'(1) = 0, and f(0) = 0, f(1) = 1 => f(t) = -2t^3+3t^2 = t^2 (3-2t)
        # NOTE: I've merely rediscovered the Smoothstep function S_1(x): https://en.wikipedia.org/wiki/Smoothstep
        t_array = t_array ** 2 * (3 - 2 * t_array)  # One line thanks to NumPy arrays
    # TODO: this might be possible to optimize by using the fact they're numpy arrays, but haven't found a nice way yet
    funcs_dict = {'linear': lerp, 'spherical': slerp}
    vectors = np.array([funcs_dict[interp_type](t, v0, v1) for t in t_array], dtype=np.float32)
    return vectors


# ----------------------------------------------------------------------------


def double_slowdown(latents: np.ndarray, duration: float, frames: int) -> Tuple[np.ndarray, float, int]:
    """
    Auxiliary function to slow down the video by 2x. We return the new latents, duration, and frames of the video
    """
    # Make an empty latent vector with double the amount of frames, but keep the others the same
    z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)
    # In the even frames, populate it with the latents
    for i in range(len(latents)):
        z[2 * i] = latents[i]
    # Interpolate in the odd frames
    for i in range(1, len(z), 2):
        # slerp between (t=0.5) even frames; for the last frame, we loop to the first one (z[0])
        z[i] = slerp(0.5, z[i - 1], z[i + 1]) if i != len(z) - 1 else slerp(0.5, z[0], z[i - 1])
    # TODO: we could change this to any slowdown: slerp(1/slowdown, ...), and we return z, slowdown * duration, ...
    # Return the new latents, and the respective new duration and number of frames
    return z, 2 * duration, 2 * frames


def global_pulsate_psi(psi_start: float, psi_end: float, n_steps: int, frequency: float = 1.0) -> torch.Tensor:
    """
    Pulsate the truncation psi parameter between start and end, taking n_steps on a sinusoidal wave.
    """
    alpha = (psi_start + psi_end) / (psi_start - psi_end)
    beta = 2 / (psi_start - psi_end)

    total_time = 2 * np.pi  # This value doesn't matter in the end

    timesteps = torch.arange(0, total_time, total_time / n_steps)
    truncation_psi = (torch.cos(frequency * timesteps) + alpha) / beta

    return truncation_psi


def wave_pulse_truncation_psi(psi_start: float,
                              psi_end: float,
                              n_steps: int,
                              grid_shape: Tuple[int, int],
                              frequency: int,
                              time: int) -> torch.Tensor:  # Output shape: [num_grid_cells, 1, 1]
    """
    Pulsate the truncation psi parameter between start and end, taking n_steps on a sinusoidal wave on a grid
        Note: The output shape should be [math.prod(grid_shape), 1, 1]
    """
    # Let's save some headaches, shall we?
    if psi_start == psi_end:
        import math
        return torch.ones(math.prod(grid_shape), 1, 1) * psi_start

    # We define a total time, but note it's specific to our definition of the wave below (the 2*pi in the conditions)
    total_time = 5 * torch.pi  # T
    timesteps = torch.arange(0, total_time, total_time / n_steps)

    # Envolope function
    def envelope(time):
        """ Envelope function that will regulate the amplitude of the wave; usage: envelope(time) * wave(time) """
        # Example: a 1D Gabor filter
        # gaussian = torch.exp(-(time - total_time / 2) ** 2 / 16)
        # sinusoid = torch.exp(1j * torch.pi(time - total_time / 2) / 2)
        # return torch.sin(time * torch.pi / total_time) / 2 + 0.5
        return torch.tensor(1.0)

    # Define the grid itself as a 2D grid where we will evaluate our wave function/psi
    width, height = grid_shape
    xs = torch.arange(0, 2*torch.pi, 2*torch.pi/width)
    ys = torch.arange(0, 2*torch.pi, 2*torch.pi/height)
    x, y = torch.meshgrid(xs, ys, indexing='xy')

    # Define the wave equation (go crazy here!)
    # In my case, I will use a sinusoidal wave with source at the upper-left corner of the grid
    # The wave will travel radially from the source, and will be truncated at the edges of the grid with the psi_start value
    r = torch.sqrt(x ** 2 + y ** 2)

    # The wave function is defined by parts, that is, keep it constant (psi_start) before and after the wave; its
    # general shape in 1D will be psi(x, t) = (cos(f(x-t)) + alpha) / beta, where alpha and beta are defined so as to
    # satisfy the boundary conditions (psi(x, 0) = psi_start, psi(x, T/2) = psi_end, psi(x, T) = psi_start))
    alpha = (psi_start + psi_end) / (psi_start - psi_end)
    beta = 2 / (psi_start - psi_end)

    def truncate(value):
        """
        Auxiliary function to interpolate between your start and end psi. Use to translate from "value=0" (psi_start)
        to "value=1" (psi_end) """
        return psi_start + value * (psi_end - psi_start)

    # Define the wave function by parts, that is, keep it constant (psi_start) before and after the wave
    truncation_psi = torch.where(torch.gt(r, timesteps[time]) | torch.lt(r, timesteps[time] - 2 * torch.pi),
                                 torch.tensor(psi_start),
                                 (torch.cos(frequency * (r - timesteps[time])) + alpha) / beta)

    # Make sure the output is of the right shape
    truncation_psi = truncation_psi.view(width*height, 1, 1)
    return truncation_psi


# ----------------------------------------------------------------------------


def make_affine_transform(m: Union[torch.Tensor, np.ndarray] = None,
                          angle: float = 0.0,
                          translate_x: float = 0.0,
                          translate_y: float = 0.0,
                          scale_x: float = 1.0,
                          scale_y: float = 1.0,
                          shear_x: float = 0.0,
                          shear_y: float = 0.0,
                          mirror_x: bool = False,
                          mirror_y: bool = False) -> np.array:
    """Make affine transformation with the given parameters. If none are passed, will return the identity.
    As a guide for affine transformations: https://en.wikipedia.org/wiki/Affine_transformation"""
    # m is the starting affine transformation matrix (e.g., G.synthesis.input.transform)
    if m is None:
        m = np.eye(3, dtype=np.float64)
    elif isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    elif isinstance(m, np.ndarray):
        pass
    # Remember these are the inverse transformations!
    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                [-np.sin(angle), np.cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
    # Translation matrix
    translation_matrix = np.array([[1.0, 0.0, -translate_x],
                                   [0.0, 1.0, -translate_y],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)
    # Scale matrix (don't let it go into negative or 0)
    scale_matrix = np.array([[1. / max(scale_x, 1e-4), 0.0, 0.0],
                             [0.0, 1. / max(scale_y, 1e-4), 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    # Shear matrix
    shear_matrix = np.array([[1.0, -shear_x, 0.0],
                             [-shear_y, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    # Mirror/reflection in x matrix
    xmirror_matrix = np.array([[1.0 - 2 * mirror_x, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
    # Mirror/reflection in y matrix
    ymirror_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0 - 2 * mirror_y, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)

    # Make the resulting affine transformation (note that these are non-commutative, so we *choose* this order)
    m = m @ rotation_matrix @ translation_matrix @ scale_matrix @ shear_matrix @ xmirror_matrix @ ymirror_matrix
    return m


def anchor_latent_space(G) -> None:
    # Thanks to @RiversHaveWings and @nshepperd1
    if hasattr(G.synthesis, 'input'):
        # Unconditional models differ by a bit
        if G.c_dim == 0:
            shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0)).squeeze(0)
        else:
            shift = G.synthesis.input.affine(G.mapping.w_avg).mean(0)
        G.synthesis.input.affine.bias.data.add_(shift)
        G.synthesis.input.affine.weight.data.zero_()


def force_fp32(G) -> None:
    """Force fp32 as in during training"""
    G.synthesis.num_fp16_res = 0
    for name, layer in G.synthesis.named_modules():
        if hasattr(layer, 'conv_clamp'):
            layer.conv_clamp = None
            layer.use_fp16 = False


def use_cpu(G) -> None:
    """Use the CPU instead of the GPU; force_fp32 must be set to True, apart from the device setting"""
    # @nurpax found this before: https://github.com/NVlabs/stylegan2-ada-pytorch/issues/54#issuecomment-793713965, but we
    # will use @JCBrouwer's solution:  https://github.com/NVlabs/stylegan2-ada-pytorch/issues/105#issuecomment-838577639
    import functools
    G.forward = functools.partial(G.forward, force_fp32=True)

# ----------------------------------------------------------------------------

resume_specs = {
        # For StyleGAN2/ADA models; --cfg=stylegan2
        'stylegan2': {
            # Official NVIDIA models
            'ffhq256':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'ffhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'celebahq256':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'lsundog256':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'afhqcat512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'afhqdog512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'afhqwild512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'brecahad512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'cifar10':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl',
            # Other configs are available at: https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/, but I will list here the config-f only
            'lsuncar512':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl',  # config-f
            'lsuncat256':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl',  # config-f
            'lsunchurch256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl',  # config-f
            'lsunhorse256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl',  # config-f
            # Community models. More can be found at: https://github.com/justinpinkney/awesome-pretrained-stylegan2 by @justinpinkney, but weren't added here
            'minecraft1024': 'https://github.com/jeffheaton/pretrained-gan-minecraft/releases/download/v1/minecraft-gan-2020-12-22.pkl',  # Thanks to @jeffheaton
            'imagenet512':   'https://battle.shawwn.com/sdc/stylegan2-imagenet-512/model.ckpt-533504.pkl',  # Thanks to @shawwn
            'wikiart1024-C': 'https://archive.org/download/wikiart-stylegan2-conditional-model/WikiArt5.pkl',  # Thanks to @pbaylies; conditional (167 classes in total: --class=0 to 166)
            'wikiart1024-U': 'https://archive.org/download/wikiart-stylegan2-conditional-model/WikiArt_Uncond2.pkl',  # Thanks to @pbaylies; unconditional
            'maps1024':      'https://archive.org/download/mapdreamer/mapdreamer.pkl',  # Thanks to @tjukanov
            'fursona512':    'https://thisfursonadoesnotexist.com/model/network-e621-r-512-3194880.pkl',  # Thanks to @arfafax
            'mlpony512':     'https://thisponydoesnotexist.net/model/network-ponies-1024-151552.pkl',  # Thanks to @arfafax
            'lhq1024':       'https://huggingface.co/justinpinkney/lhq-sg2-1024/resolve/main/lhq-sg2-1024.pkl',  # Thanks to @justinpinkney
            # Deceive-D/APA models (ignoring the faces models): https://github.com/EndlessSora/DeceiveD
            'afhqcat256':    'https://drive.google.com/u/0/uc?export=download&confirm=zFoN&id=1P9ouHIK-W8JTb6bvecfBe4c_3w6gmMJK',
            'anime256':      'https://drive.google.com/u/0/uc?export=download&confirm=6Uie&id=1EWOdieqELYmd2xRxUR4gnx7G10YI5dyP',
            'cub256':        'https://drive.google.com/u/0/uc?export=download&confirm=KwZS&id=1J0qactT55ofAvzddDE_xnJEY8s3vbo1_',
            # Self-Distilled StyleGAN (full body representation of each class): https://github.com/self-distilled-stylegan/self-distilled-internet-photos
            'sddogs1024':    'https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl',
            'sdelephant512': 'https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl',
            'sdhorses256':   'https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl',
            'sdbicycles256': 'https://storage.googleapis.com/self-distilled-stylegan/bicycles_256_pytorch.pkl',
            'sdlions512':    'https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl',
            'sdgiraffes512': 'https://storage.googleapis.com/self-distilled-stylegan/giraffes_512_pytorch.pkl',
            'sdparrots512':  'https://storage.googleapis.com/self-distilled-stylegan/parrots_512_pytorch.pkl'

        },
        # For StyleGAN2 extended (--cfg=styelgan2-ext)
        'stylegan2-ext': {
            'anime512': 'https://drive.google.com/u/0/uc?export=download&confirm=zFoN&id=1A-E_E32WAtTHRlOzjhhYhyyBDXLJN9_H'  # Thanks to @aydao
        },
        # For StyleGAN3 config-r models (--cfg=stylegan3-r)
        'stylegan3-r': {
            # Official NVIDIA models
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
        },
        # For StyleGAN3 config-t models (--cfg=stylegan3-t)
        'stylegan3-t': {
            # Official NVIDIA models
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            # Community models, Wombo Dream-based models found in: https://github.com/edstoica/lucid_stylegan3_datasets_models by @edstoica; TODO: more to come, update the list as they are released!
            'mechfuture256': 'https://www.dropbox.com/s/v2oie53cz62ozvu/network-snapshot-000029.pkl?dl=1',  # Thanks to @edstoica; 29kimg tick
            'vivflowers256': 'https://www.dropbox.com/s/o33lhgnk91hstvx/network-snapshot-000069.pkl?dl=1',  # Thanks to @edstoica; 68kimg tick
            'alienglass256': 'https://www.dropbox.com/s/gur14k0e7kspguy/network-snapshot-000038.pkl?dl=1',  # Thanks to @edstoica; 38kimg tick
            'scificity256': 'https://www.dropbox.com/s/1kfsmlct4mriphc/network-snapshot-000210.pkl?dl=1',  # Thanks to @edstoica; 210kimg tick
            'scifiship256': 'https://www.dropbox.com/s/02br3mjkma1hubc/network-snapshot-000162.pkl?dl=1',  # Thanks to @edstoica; 168kimg tick
        }
}

# ----------------------------------------------------------------------------


# TODO: all of the following functions must work for RGBA images
def w_to_img(G, dlatents: Union[List[torch.Tensor], torch.Tensor],
             noise_mode: str = 'const',
             new_w_avg: torch.Tensor = None,
             truncation_psi: float = 1.0) -> np.ndarray:
    """
    Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
    returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        Note: this function should be used after doing the truncation trick!
        Note: Optionally, you can also pass a new_w_avg to use instead of the one in G, with a reverse
              truncation trick
    """
    # If we have a single dlatent, we need to add a batch dimension
    assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
    if len(dlatents.shape) == 2:
        dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
    if new_w_avg is not None:
        new_w_avg = new_w_avg.to(next(G.parameters()).device)
        dlatents = (dlatents - new_w_avg) * (1 - truncation_psi) + new_w_avg
    synth_image = G.synthesis(dlatents, noise_mode=noise_mode)
    synth_image = (synth_image + 1) * 255/2  # [-1.0, 1.0] -> [0.0, 255.0]
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
    return synth_image


def z_to_dlatent(G, latents: torch.Tensor, label: torch.Tensor, truncation_psi: float = 1.0) -> torch.Tensor:
    """Get the dlatent from the given latent, class label and truncation psi"""
    assert isinstance(latents, torch.Tensor), f'latents should be a torch.Tensor!: "{type(latents)}"'
    assert isinstance(label, torch.Tensor), f'label should be a torch.Tensor!: "{type(label)}"'
    if len(latents.shape) == 1:
        latents = latents.unsqueeze(0)  # An individual latent => [1, G.z_dim]
    dlatents = G.mapping(z=latents, c=label, truncation_psi=truncation_psi)

    return dlatents


def z_to_img(G, latents: torch.Tensor, label: torch.Tensor, truncation_psi: float, noise_mode: str = 'const') -> np.ndarray:
    """
    Get an image/np.ndarray from a latent Z using G, the label, truncation_psi, and noise_mode. The shape
    of the output image/np.ndarray will be [len(latents), G.img_resolution, G.img_resolution, G.img_channels]
    """
    dlatents = z_to_dlatent(G=G, latents=latents, label=label, truncation_psi=1.0)
    dlatents = G.mapping.w_avg + (G.mapping.w_avg - dlatents) * truncation_psi
    img = w_to_img(G=G, dlatents=dlatents, noise_mode=noise_mode)  # Let's not redo code
    return img


def get_w_from_seed(G, device: Union[str, torch.device], seed: int, truncation_psi: float, new_w_avg: torch.Tensor = None) -> torch.Tensor:
    """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
    z = np.random.RandomState(seed).randn(1, G.z_dim)
    w = G.mapping(torch.from_numpy(z).to(device), None)
    w_avg = G.mapping.w_avg if new_w_avg is None else new_w_avg.to(device)
    w = w_avg + (w - w_avg) * truncation_psi

    return w


def get_latent_from_file(file: Union[str, os.PathLike],
                    return_ext: bool = False,
                    named_latent: str = 'w') -> Tuple[np.ndarray, Optional[str]]:
    """Get dlatent (w) from a .npy or .npz file"""
    filename, file_extension = os.path.splitext(file)
    assert file_extension in ['.npy', '.npz'], f'"{file}" has wrong file format! Only ".npy" or ".npz" are allowed'
    if file_extension == '.npy':
        latent = np.load(file)
        extension = '.npy'
    else:
        latent = np.load(file)[named_latent]
        extension = '.npz'
    if len(latent.shape) == 4:
        latent = latent[0]
    return (latent, extension) if return_ext else latent


# ----------------------------------------------------------------------------


def save_config(ctx: click.Context, run_dir: Union[str, os.PathLike], save_name: str = 'config.json') -> None:
    """Save the configuration stored in ctx.obj into a JSON file at the output directory."""
    with open(os.path.join(run_dir, save_name), 'w') as f:
        json.dump(ctx.obj, f, indent=4, sort_keys=True)


# ----------------------------------------------------------------------------


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir
