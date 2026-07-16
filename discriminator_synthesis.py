import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from torchvision import transforms

import PIL
from PIL import Image, ImageColor

import numpy as np

import os
import shutil
import subprocess
import click
from dataclasses import dataclass
from typing import Union, Tuple, Optional, List, Type
from tqdm import tqdm
import re

from torch_utils import gen_utils
from network_features import DiscriminatorFeatures
from fractalperlin import get_2d_perlin, get_3d_perlin
from pytorch_ssim import gaussian as _gaussian_1d


# ----------------------------------------------------------------------------


@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


def get_available_layers(max_resolution: int) -> List[str]:
    """Helper function to get the available layers given a max resolution (first block in the Discriminator)"""
    max_res_log2 = int(np.log2(max_resolution))
    block_resolutions = [2**i for i in range(max_res_log2, 2, -1)]

    available_layers = ['from_rgb']
    for block_res in block_resolutions:
        # We don't add the skip layer, as it's the same as conv1 (due to in-place addition; could be changed)
        available_layers.extend([f'b{block_res}_conv0', f'b{block_res}_conv1'])
    # We also skip 'b4_mbstd', as it doesn't add any new information compared to b8_conv1
    available_layers.extend(['b4_conv', 'fc', 'out'])
    return available_layers


# ----------------------------------------------------------------------------


def get_model_name(network_pkl: Union[str, os.PathLike], cfg: Optional[str] = None, max_length: int = 10) -> str:
    """
    Compact model name for directory/file naming. Known model names from gen_utils.resume_specs
    (e.g., 'ffhq1024', 'metfaces1024') are used in full; otherwise the file name of the local
    path or URL (whatever precedes '.pkl') is used, shortened to max_length characters.
    """
    network_pkl = str(network_pkl)
    if cfg is not None and network_pkl in gen_utils.resume_specs.get(cfg, {}):
        return network_pkl
    name = network_pkl.replace('\\', '/').split('/')[-1].split('.pkl')[0]
    return name[:max_length] if len(name) > max_length else name


def get_run_desc(network_pkl: Union[str, os.PathLike], cfg: Optional[str], *parts: str) -> str:
    """Build a run directory description: '<model_name>[-<cfg>]-<part>-<part>-...'"""
    return '-'.join([get_model_name(network_pkl, cfg)] + ([cfg] if cfg is not None else []) + [p for p in parts if p])


def layers_tag(layers: List[str], max_layers: int = 3) -> str:
    """Compact tag describing the layers used, for directory naming"""
    return '-'.join(layers) if len(layers) <= max_layers else f'{len(layers)}_layers'


# ----------------------------------------------------------------------------
# Video helpers: call the ffmpeg binary directly via subprocess


def _ffmpeg_bin() -> str:
    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        raise RuntimeError('ffmpeg not found! Install it and make sure it is in your PATH.')
    return ffmpeg_bin


def save_video_ffmpeg(frames_dir: Union[str, os.PathLike],
                      image_pattern: str,
                      video_name: str,
                      fps: int = 30,
                      crf: int = 20,
                      preset: str = 'veryfast',
                      reverse_video: bool = False) -> str:
    """
    Assemble the frames matching `image_pattern` (printf-style, e.g. 'frame_%04d.jpg') in
    `frames_dir` into an .mp4, by invoking the ffmpeg binary directly. Returns the video path.
    """
    video_path = os.path.join(frames_dir, f'{video_name}.mp4')
    print(f'Saving video to "{video_path}"...')
    subprocess.run([_ffmpeg_bin(), '-y', '-loglevel', 'error',
                    '-framerate', str(fps), '-i', os.path.join(frames_dir, image_pattern),
                    '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
                    video_path], check=True)
    if reverse_video:
        # Save the reversed video apart from the original one, so the user can compare both
        subprocess.run([_ffmpeg_bin(), '-y', '-loglevel', 'error', '-i', video_path,
                        '-vf', 'reverse', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                        '-pix_fmt', 'yuv420p', os.path.join(frames_dir, f'{video_name}_reversed.mp4')], check=True)
    return video_path


# ----------------------------------------------------------------------------
# DeepDream code; modified from Erik Linder-Norén's repository: https://github.com/eriklindernoren/PyTorch-Deep-Dream

def get_image(seed: int = 0,
              image_noise: str = 'random',
              starting_image: Union[str, os.PathLike] = None,
              image_size: int = 1024,
              convert_to_grayscale: bool = False,
              device: torch.device = torch.device('cpu')) -> Tuple[PIL.Image.Image, str]:
    """
    Get or generate an image for DeepDream synthesis.

    Args:
        seed: Random seed for reproducibility
        image_noise: 'random', 'perlin', or any solid color understood by PIL
                     (a name like 'white'/'black'/'red', or hex like '#ff8800')
        starting_image: Path to existing image (if None, generates new)
        image_size: Size of generated image
        convert_to_grayscale: Convert to grayscale
        device: Device for noise generation

    Returns:
        (PIL Image, filename string)
    """
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    # Load existing image if provided
    if starting_image is not None:
        image = Image.open(starting_image).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    else:
        if image_noise == 'random':
            starting_image = f'random_image-seed_{seed:08d}.jpg'
            image = Image.fromarray(rnd.randint(0, 255, (image_size, image_size, 3), dtype='uint8'))
        elif image_noise == 'perlin':
            starting_image = f'perlin_image-seed_{seed:08d}.jpg'
            # Use our local fractalperlin implementation; one independent noise field per RGB channel
            noise = get_2d_perlin((3, image_size, image_size), seed=seed, device=device, octaves=6)
            noise = noise.cpu().numpy()
            # Stretch to the full [0, 255] range (raw fractal Perlin rarely reaches +-1)
            noise = (noise - noise.min()) / (np.ptp(noise) + 1e-8)
            image = Image.fromarray((255 * noise).astype(np.uint8).transpose(1, 2, 0))
        else:
            # Treat anything else as a solid color (e.g., 'white', 'black', 'red', '#ff8800')
            try:
                color = ImageColor.getrgb(image_noise)
            except ValueError:
                raise ValueError(f'Unknown image noise type/color "{image_noise}"! Use "random", "perlin", '
                                 'or any color name/hex code understood by PIL (e.g., "white", "#ff8800")')
            starting_image = f'{image_noise.replace("#", "hex")}_image.jpg'
            image = Image.new('RGB', (image_size, image_size), color)

    if convert_to_grayscale:
        image = image.convert('L').convert('RGB')

    return image, starting_image


def get_perlin_volume(seed: int,
                      num_frames: int,
                      image_size: int,
                      convert_to_grayscale: bool = False,
                      device: torch.device = torch.device('cuda'),
                      loop: bool = True,
                      octaves: int = 6) -> np.ndarray:
    """
    Generate a 3D fractal Perlin noise volume that can be sliced along any axis.

    Args:
        seed: Random seed
        num_frames: Size of the volume along the time axis
        image_size: Size of the volume along the y and x axes
        convert_to_grayscale: Use a single noise field for all three channels
        device: Device for generation (the volume is returned on CPU)
        loop: If True, the volume is periodic along the time axis (seamless loop)
        octaves: Number of fractal octaves to sum

    Returns:
        uint8 array of shape (num_frames, image_size, image_size, 3); the min/max
        are taken over the whole volume so slices stay temporally consistent
    """
    channels = 1 if convert_to_grayscale else 3
    noise = get_3d_perlin((channels, image_size, image_size), num_frames, seed=seed,
                          device=device, octaves=octaves, loop=loop)  # (T, C, H, W)
    noise = noise.cpu().numpy()
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-8)
    volume = (255 * noise).astype(np.uint8).transpose(0, 2, 3, 1)  # (T, H, W, C)
    if convert_to_grayscale:
        volume = volume.repeat(3, axis=-1)
    return volume


def get_padding_multiple(layers: List[str], max_resolution: int) -> int:
    """
    Smallest power of two that each spatial dimension of an input image must be divisible by,
    so that the requested Discriminator layers can be computed on inputs of arbitrary size
    (every block halves the resolution, and the skip/conv branches must stay in sync).
    """
    max_down = 0
    for layer in layers:
        match = re.match(r'b(\d+)_conv(\d)', layer)
        if layer == 'from_rgb':
            down = 0
        elif match and int(match.group(1)) > 4:
            res = int(match.group(1))
            down = int(np.log2(max_resolution // res)) + (1 if match.group(2) == '1' else 0)
        else:  # 'b4_mbstd', 'b4_conv', 'fc', 'out' traverse the full network
            down = int(np.log2(max_resolution // 4))
        max_down = max(max_down, down)
    return 2 ** max_down


def pad_to_multiple(image: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Tile a (C, H, W) or (B, C, H, W) tensor along its bottom/right edges so both spatial
    dimensions are multiples of `multiple`. Tiling (rather than zero-padding) keeps the
    statistics of the noise, and is seamless along a looping Perlin time axis.

    Returns the padded tensor and the original (H, W) to crop the result back.
    """
    h, w = image.shape[-2:]
    new_h = -(-h // multiple) * multiple
    new_w = -(-w // multiple) * multiple
    if (new_h, new_w) != (h, w):
        reps = [1] * image.dim()
        reps[-2], reps[-1] = -(-new_h // h), -(-new_w // w)
        image = image.repeat(*reps)[..., :new_h, :new_w]
    return image, (h, w)


def crop_resize_rotate(img: PIL.Image.Image,
                       crop_size: int = None,
                       new_size: int = None,
                       rotation_deg: float = None,
                       translate_x: float = 0.0,
                       translate_y: float = 0.0) -> PIL.Image.Image:
    """Center-crop the input image into a square of sides crop_size; can be resized to new_size; rotated rotation_deg counter-clockwise"""
    # Center-crop the input image
    if crop_size is not None:
        w, h = img.size                                         # Input image width and height
        img = img.crop(box=((w - crop_size) // 2,               # Left pixel coordinate
                            (h - crop_size) // 2,               # Upper pixel coordinate
                            (w + crop_size) // 2,               # Right pixel coordinate
                            (h + crop_size) // 2))              # Lower pixel coordinate
    # Resize
    if new_size is not None:
        img = img.resize(size=(new_size, new_size),             # Requested size of the image in pixels; (width, height)
                         resample=Image.LANCZOS)                # Resampling filter
    # Rotation and translation
    if rotation_deg is not None:
        img = img.rotate(angle=rotation_deg,                    # Angle to rotate image, counter-clockwise
                         resample=Image.BICUBIC,                # Resampling filter; options: Image.Resampling.{NEAREST, BILINEAR, BICUBIC}
                         expand=False,                          # If True, the whole rotated image will be shown
                         translate=(translate_x, translate_y),  # Translate the image, from top-left corner (post-rotation)
                         fillcolor=(0, 0, 0))                   # Black background
    # TODO: tile the background
    return img


# StyleGAN's Discriminator is fed images in [-1, 1], so we normalize with mean = std = 0.5
# (the previous ImageNet statistics are a VGG/DeepDream legacy and don't match D's training data)
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Per-channel bounds in normalized space (the [0, 1] image range maps to these)
_clip_min = torch.as_tensor((0.0 - mean) / std, dtype=torch.float32).view(1, -1, 1, 1)
_clip_max = torch.as_tensor((1.0 - mean) / std, dtype=torch.float32).view(1, -1, 1, 1)


def deprocess(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Undo the preprocessing normalization; returns a uint8 HWC image,
    or a BHWC stack if a batch of more than one image is given"""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    image = image.transpose(0, 2, 3, 1) if image.ndim == 4 else image.transpose(1, 2, 0)
    image = image * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image = np.clip(image, 0.0, 1.0)
    return (255 * image).astype('uint8')


def clip(image_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp per channel to the valid (normalized) image range; vectorized, stays on device"""
    lo = _clip_min.to(image_tensor.device)
    hi = _clip_max.to(image_tensor.device)
    return torch.min(torch.max(image_tensor, lo), hi)


# ----------------------------------------------------------------------------
# Anti-saturation helpers. DeepDream ascent on the feature norm, bounded only by a hard clamp,
# drifts to pure white and pins there. These keep brightness/contrast steady and remove the
# hard 0/1 pile-up. See the DreamConfig dataclass for the knobs and get_image for starting colors.


@dataclass
class DreamConfig:
    """Anti-saturation settings threaded through deep_dream()/dream()."""
    anti_saturation: bool = True         # renormalize to a held mean/std each step
    soft_clip: bool = True               # smooth (tanh-family) bound instead of hard clamp
    target_mean: Optional[float] = None  # [0, 1]; None -> hold the starting image's own mean
    target_std: Optional[float] = None   # [0, 1]; None -> hold the starting image's own std
    grad_smooth: float = 0.5             # Gaussian sigma to blur the gradient (0 = off)
    jitter: int = 0                      # max random shift in pixels for the forward pass (0 = off)
    flip: bool = False                   # random horizontal flip for the forward pass


def soft_clip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0, beta: float = 8.0) -> torch.Tensor:
    """
    Softplus-based smooth clamp to [lo, hi]: near-identity in the interior, easing smoothly past the
    bounds without ever reaching them exactly. Preferred over plain tanh(x), which compresses interior
    contrast every step and would fight the renormalization into a washed-out result.
    """
    x = lo + F.softplus(x - lo, beta=beta)
    x = hi - F.softplus(hi - x, beta=beta)
    return x


_gauss_window_cache = {}


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur of a (B, C, H, W) tensor; reuses pytorch_ssim's 1D Gaussian."""
    if sigma <= 0:
        return x
    channels = x.shape[1]
    key = (round(sigma, 3), channels, x.device, x.dtype)
    window = _gauss_window_cache.get(key)
    if window is None:
        radius = max(1, int(round(3 * sigma)))
        window_size = 2 * radius + 1
        w1d = _gaussian_1d(window_size, sigma).to(device=x.device, dtype=x.dtype)
        w2d = torch.outer(w1d, w1d)
        window = w2d.expand(channels, 1, window_size, window_size).contiguous()
        _gauss_window_cache[key] = window
    padding = window.shape[-1] // 2
    return F.conv2d(x, window, padding=padding, groups=channels)


def _resolve_target(image: torch.Tensor, config: DreamConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve the per-sample, per-channel (mean, std) that renormalization holds the image to.
    Defaults to the input image's own statistics (preserving the starting color/brightness);
    a flat solid-color start (std ~ 0) falls back to a neutral std so structure can still emerge.
    Explicit config.target_mean/std are given in [0, 1] and converted to the normalized [-1, 1] space.
    """
    neutral_std_norm = 2 * 0.22  # 0.22 in [0, 1] -> normalized space (std scales by 1/0.5)
    t_mean = image.mean(dim=(2, 3), keepdim=True)
    t_std = image.std(dim=(2, 3), keepdim=True)
    t_std = torch.where(t_std < 1e-3, torch.full_like(t_std, neutral_std_norm), t_std)
    if config.target_mean is not None:
        t_mean = torch.full_like(t_mean, 2 * config.target_mean - 1)
    if config.target_std is not None:
        t_std = torch.full_like(t_std, 2 * config.target_std)
    return t_mean, t_std


def renormalize(image: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor,
                eps: float = 1e-5) -> torch.Tensor:
    """Per-sample, per-channel standardize the image back to (target_mean, target_std)."""
    cur_mean = image.mean(dim=(2, 3), keepdim=True)
    cur_std = image.std(dim=(2, 3), keepdim=True)
    return (image - cur_mean) / (cur_std + eps) * target_std + target_mean


def dream(image: torch.Tensor,
          model: torch.nn.Module,
          layers: List[str],
          channels: List[int] = None,
          normed: bool = False,
          sqrt_normed: bool = False,
          iterations: int = 20,
          lr: float = 1e-2,
          config: DreamConfig = None,
          target_mean: torch.Tensor = None,
          target_std: torch.Tensor = None,
          generator: torch.Generator = None) -> torch.Tensor:
    """
    Updates the (preprocessed, on-device) image to maximize the chosen layer outputs for n iterations.
    Batches are supported: the loss and the gradient-normalized step are computed per sample,
    so each image in the batch dreams independently, exactly as if processed alone.

    Anti-saturation (via `config`): optional per-forward jitter/flip augmentation, Gaussian blur of
    the gradient (high-frequency suppression), renormalization to a held mean/std, and a smooth bound
    replacing the hard clamp. With `config.anti_saturation` off and no blur/jitter it reduces exactly
    to the original hard-clip ascent.
    """
    config = config if config is not None else DreamConfig()
    if config.anti_saturation and target_mean is None:
        target_mean, target_std = _resolve_target(image, config)

    image = image.detach().clone().requires_grad_(True)
    for _ in range(iterations):
        # Optional augmentation for the forward pass only; autograd carries the gradient back to
        # `image` through the (differentiable) roll/flip, so no manual un-shift is needed.
        inp = image
        if config.jitter > 0:
            sy, sx = torch.randint(-config.jitter, config.jitter + 1, (2,), generator=generator).tolist()
            inp = torch.roll(inp, shifts=(sy, sx), dims=(2, 3))
        if config.flip and torch.rand(1, generator=generator).item() < 0.5:
            inp = torch.flip(inp, dims=[3])

        out = model.get_layers_features(inp, layers=layers, channels=channels, normed=normed, sqrt_normed=sqrt_normed)
        loss = sum(layer.flatten(1).norm(dim=1).sum() for layer in out)  # Per-sample norm; more than one layer may be used
        loss.backward()
        with torch.no_grad():
            grad = image.grad
            if config.grad_smooth > 0:
                grad = gaussian_blur(grad, config.grad_smooth)
            avg_grad = grad.abs().mean(dim=(1, 2, 3), keepdim=True)
            image += lr / (avg_grad + 1e-12) * grad
            if config.anti_saturation:
                image.copy_(renormalize(image, target_mean, target_std))
                if config.soft_clip:
                    image.copy_(soft_clip(image))
                else:
                    image.copy_(clip(image))
            else:
                image.copy_(clip(image))
            image.grad.zero_()
    return image.detach()


def deep_dream(image: Union[PIL.Image.Image, torch.Tensor],
               model: torch.nn.Module,
               model_resolution: int,
               layers: List[str],
               channels: List[int],
               seed: Union[int, Type[None]],
               normed: bool,
               sqrt_normed: bool,
               iterations: int,
               lr: float,
               octave_scale: float,
               num_octaves: int,
               unzoom_octave: bool = False,
               disable_inner_tqdm: bool = False,
               ignore_initial_transform: bool = False,
               config: DreamConfig = None) -> np.ndarray:
    """
    Main deep dream method. `image` can be a PIL.Image (which will be preprocessed) or an
    already-preprocessed tensor of shape (C, H, W) or (1, C, H, W) in normalized [-1, 1] space.
    Everything runs on the model's device; the only CPU transfer is the final deprocessed result.

    Anti-saturation settings are carried in `config` (see DreamConfig); the mean/std target is
    captured once from the full-resolution input and held across every octave, so global
    brightness/contrast stays locked to the starting image.
    """
    config = config if config is not None else DreamConfig()
    device = next(model.parameters()).device
    if isinstance(image, Image.Image):
        # Center-crop and resize
        if not ignore_initial_transform:
            image = crop_resize_rotate(img=image, crop_size=min(image.size), new_size=model_resolution)
        image = preprocess(image)
    image = image.detach().to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Capture the held mean/std target once, from the full-res input (global brightness/contrast lock)
    target_mean = target_std = None
    if config.anti_saturation:
        target_mean, target_std = _resolve_target(image, config)

    # Seed a generator for reproducible jitter/flip augmentation (if enabled). `seed` may be an int,
    # a list of seeds (batched dream), or None; derive a single int to seed the generator.
    aug_generator = None
    if config.jitter > 0 or config.flip:
        if isinstance(seed, (list, tuple)):
            seed_val = int(seed[0]) if len(seed) else 0
        elif seed is not None:
            seed_val = int(seed)
        else:
            seed_val = 0
        aug_generator = torch.Generator(device='cpu').manual_seed(seed_val)

    # Extract image representations for each octave (coarse to fine), on-device
    octaves = [image]
    for _ in range(num_octaves - 1):
        prev = octaves[-1]
        h, w = prev.shape[-2:]
        new_size = (max(int(h / octave_scale), 8), max(int(w / octave_scale), 8))
        octave = F.interpolate(prev, size=new_size, mode='bilinear', align_corners=False)
        # Necessary for StyleGAN's Discriminator, as it cannot handle any image size
        if unzoom_octave:
            octave = F.interpolate(octave, size=(h, w), mode='bilinear', align_corners=False)
        octaves.append(octave)

    detail = torch.zeros_like(octaves[-1])
    tqdm_desc = f'Dreaming w/layers {"|".join(x for x in layers)}'
    tqdm_desc = f'Seed: {seed} - {tqdm_desc}' if seed is not None else tqdm_desc
    for octave_idx, octave_base in enumerate(tqdm(octaves[::-1], desc=tqdm_desc, disable=disable_inner_tqdm)):
        if octave_idx > 0:
            # Upsample detail to new octave dimension
            detail = F.interpolate(detail, size=octave_base.shape[-2:], mode='bilinear', align_corners=False)
        # Add deep dream detail from previous octave to new base, and get new deep dream image
        dreamed_image = dream(octave_base + detail, model, layers, channels, normed, sqrt_normed, iterations, lr,
                              config=config, target_mean=target_mean, target_std=target_std, generator=aug_generator)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


# ----------------------------------------------------------------------------


def saturation_options(default_jitter: int = 0, default_flip: bool = False):
    """
    Reusable set of anti-saturation click options shared by all generation commands. Jitter/flip
    are random per forward pass, so they default off for the sequence commands (per-frame randomness
    adds flicker) and on for the single-image `dream`. Returns a decorator that stacks the options.
    """
    def decorator(func):
        options = [
            click.option('--anti-saturation/--no-anti-saturation', 'anti_saturation', default=True, show_default=True,
                         help='Renormalize to a held mean/std each step so brightness/contrast stay steady (stops the drift to white)'),
            click.option('--soft-clip/--no-soft-clip', 'soft_clip', default=True, show_default=True,
                         help='Smoothly bound pixels (tanh-family) instead of a hard 0/1 clamp'),
            click.option('--target-mean', type=click.FloatRange(0.0, 1.0), default=None,
                         help='Target pixel mean in [0, 1] to hold (default: the starting image\'s own mean)'),
            click.option('--target-std', type=click.FloatRange(0.0, 1.0), default=None,
                         help='Target pixel std in [0, 1] to hold (default: the starting image\'s own std)'),
            click.option('--grad-smooth', type=click.FloatRange(min=0.0), default=0.5, show_default=True,
                         help='Gaussian sigma to blur the gradient each step, suppressing high frequencies (0 = off)'),
            click.option('--jitter', type=click.IntRange(min=0), default=default_jitter, show_default=True,
                         help='Max random shift in pixels applied to the forward pass (transformation robustness; 0 = off)'),
            click.option('--flip/--no-flip', 'flip', default=default_flip, show_default=True,
                         help='Randomly horizontally flip the forward pass (transformation robustness)'),
        ]
        for option in reversed(options):
            func = option(func)
        return func
    return decorator


def build_dream_config(anti_saturation: bool, soft_clip: bool, target_mean: Optional[float],
                       target_std: Optional[float], grad_smooth: float, jitter: int, flip: bool) -> DreamConfig:
    """Assemble a DreamConfig from the saturation_options click params."""
    return DreamConfig(anti_saturation=anti_saturation, soft_clip=soft_clip, target_mean=target_mean,
                       target_std=target_std, grad_smooth=grad_smooth, jitter=jitter, flip=flip)


# ----------------------------------------------------------------------------

# Helper functions (all base code taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # (batch_size, no. feature maps, dims of a f. map (N=c*d))

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # 'Normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)  # can also do torch.numel(input) to get the number of elements


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


@main.command(name='dream-transfer', help='Use the StyleGAN2/3 Discriminator to perform style transfer')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
@click.option('--content', type=str, help='Content image filename (url or local path)', required=True)
@click.option('--style', type=str, help='Style image filename (url or local path)', required=True)
@click.option('--content-layers', type=str, help='Comma-separated discriminator layers for content', default='b16_conv1', show_default=True)
@click.option('--style-layers', type=str, help='Comma-separated discriminator layers for style', default='b64_conv0,b32_conv0,b16_conv0,b8_conv0', show_default=True)
@click.option('--content-weight', type=float, help='Weight for content loss', default=1.0, show_default=True)
@click.option('--style-weight', type=float, help='Weight for style loss', default=1e6, show_default=True)
@click.option('--iterations', type=int, help='Number of optimization steps', default=300, show_default=True)
@click.option('--lr', type=float, help='Learning rate', default=1e-1, show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Output directory', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True)
@click.option('--description', '-desc', type=str, help='Additional description for output directory', default='', show_default=True)
def style_transfer_discriminator(
        ctx: click.Context,
        network_pkl: str,
        cfg: str,
        content: str,
        style: str,
        content_layers: str,
        style_layers: str,
        content_weight: float,
        style_weight: float,
        iterations: int,
        lr: float,
        outdir: str,
        description: str,
):
    """
    Perform neural style transfer using discriminator features.

    Optimizes an image to match the content of one image and the style of another,
    using features extracted from a StyleGAN2/3 discriminator instead of VGG.

    Reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)
    model_resolution = D.img_resolution
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Parse layers
    content_layers = content_layers.split(',')
    style_layers = style_layers.split(',')

    # Validate layers
    available_layers = get_available_layers(max_resolution=model_resolution)
    content_layers = [l for l in content_layers if l in available_layers]
    style_layers = [l for l in style_layers if l in available_layers]

    # Load and preprocess images
    def load_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((model_resolution, model_resolution), Image.LANCZOS)
        img = preprocess(img).unsqueeze(0)
        return img.to(device)

    content_img = load_image(content)
    style_img = load_image(style)

    # Start from content image (or could use noise)
    input_img = content_img.clone()
    input_img.requires_grad_(True)

    # Extract target features
    with torch.no_grad():
        content_features = model.get_layers_features(content_img, layers=content_layers)
        style_features = model.get_layers_features(style_img, layers=style_layers)

    # Create loss modules
    content_losses = [ContentLoss(feat) for feat in content_features]
    style_losses = [StyleLoss(feat) for feat in style_features]

    # Optimizer
    optimizer = torch.optim.LBFGS([input_img], lr=lr, max_iter=20)

    # Make output directory
    desc = get_run_desc(network_pkl, cfg, 'style-transfer')
    desc = f'{desc}-{description}' if description else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Save original images
    content_pil = Image.open(content).convert('RGB').resize((model_resolution, model_resolution), Image.LANCZOS)
    style_pil = Image.open(style).convert('RGB').resize((model_resolution, model_resolution), Image.LANCZOS)
    content_pil.save(os.path.join(run_dir, 'content.jpg'))
    style_pil.save(os.path.join(run_dir, 'style.jpg'))

    print(f'Running style transfer for {iterations} iterations...')
    print(f'Content layers: {content_layers}')
    print(f'Style layers: {style_layers}')

    iteration = [0]

    def closure():
        # Clamp input image
        input_img.data = clip(input_img.data)

        optimizer.zero_grad()

        # Get features from current image
        current_features_content = model.get_layers_features(input_img, layers=content_layers)
        current_features_style = model.get_layers_features(input_img, layers=style_layers)

        # Compute losses
        content_loss = 0
        for i, loss_module in enumerate(content_losses):
            content_loss += F.mse_loss(current_features_content[i], loss_module.target)

        style_loss = 0
        for i, loss_module in enumerate(style_losses):
            G_current = gram_matrix(current_features_style[i])
            style_loss += F.mse_loss(G_current, loss_module.target)

        # Weighted combination
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        iteration[0] += 1
        if iteration[0] % 50 == 0:
            print(f'Iteration {iteration[0]}/{iterations} | Content Loss: {content_loss.item():.4f} | Style Loss: {style_loss.item():.4f}')

        return total_loss

    # Optimization loop
    for _ in tqdm(range(iterations // 20), desc='Style transfer'):
        optimizer.step(closure)

    # Final image
    input_img.data = clip(input_img.data)
    output = deprocess(input_img.cpu().data.numpy())

    # Save result
    Image.fromarray(output).save(os.path.join(run_dir, 'stylized.jpg'))

    # Save configuration
    ctx.obj = {
        'network_pkl': network_pkl,
        'content_image': content,
        'style_image': style,
        'content_layers': content_layers,
        'style_layers': style_layers,
        'content_weight': content_weight,
        'style_weight': style_weight,
        'iterations': iterations,
        'lr': lr,
        'outdir': run_dir,
        'description': description
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    print(f'Style transfer complete! Results saved to {run_dir}')



# ----------------------------------------------------------------------------


@main.command(name='dream', help='Discriminator Dreaming with the StyleGAN2/3 Discriminator and the chosen layers')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='Random seeds to use. Accepted comma-separated values, ranges, or combinations: "a,b,c", "a-c", "a,b-d,e".', default='0')
@click.option('--random-image-noise', '-noise', 'image_noise', type=str, help='Starting image content: "random" noise, "perlin" noise, or any solid color understood by PIL (e.g., "white", "black", "#ff8800")', default='perlin', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=1e-2, show_default=True)
@click.option('--iterations', '-it', type=int, help='Number of gradient ascent steps per octave', default=20, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Layers of the Discriminator to use as the features. If "all", will generate a dream image per available layer in the loaded model. If "use_all", will use all available layers.', default='b16_conv1', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=int, help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=True, show_default=True)
# Performance options
@click.option('--batch-size', type=click.IntRange(min=1), help='Number of seeds to dream per forward pass (ignored when --layers=all)', default=1, show_default=True)
# Anti-saturation options (jitter/flip default on for single images)
@saturation_options(default_jitter=8, default_flip=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seeds: List[int],
        image_noise: str,
        starting_image: Union[str, os.PathLike],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # For conditional models (not yet implemented)
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: Optional[List[int]],
        norm_model_layers: bool,
        sqrt_norm_model_layers: bool,
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: bool,
        batch_size: int,
        anti_saturation: bool,
        soft_clip: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        grad_smooth: float,
        jitter: int,
        flip: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get the model resolution (image resizing and getting available layers)
    model_resolution = D.img_resolution

    # TODO: do this better, as we can combine these conditions later
    layers = layers.split(',')

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Anti-saturation configuration (shared across all seeds/layers)
    config = build_dream_config(anti_saturation, soft_clip, target_mean, target_std, grad_smooth, jitter, flip)
    anti_saturation_cfg = vars(config)

    if 'all' in layers:
        # Get all the available layers in a list
        layers = get_available_layers(max_resolution=model_resolution)

        for seed in seeds:
            # Get the image and image name
            image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                              starting_image=starting_image,
                                              image_size=model_resolution,
                                              convert_to_grayscale=convert_to_grayscale)

            # Make the run dir in the specified output directory
            desc = get_run_desc(network_pkl, cfg, 'dream-all_layers', f'seed_{seed}')
            desc = f'{desc}-{description}' if len(description) != 0 else desc
            run_dir = gen_utils.make_run_dir(outdir, desc)

            # Save starting image
            image.save(os.path.join(run_dir, f'{os.path.basename(starting_image).split(".")[0]}.jpg'))

            # Save the configuration used
            ctx.obj = {
                'network_pkl': network_pkl,
                'synthesis_options': {
                    'seed': seed,
                    'random_image_noise': image_noise,
                    'starting_image': starting_image,
                    'class_idx': class_idx,
                    'learning_rate': learning_rate,
                    'iterations': iterations},
                'layer_options': {
                    'layer': layers,
                    'channels': channels,
                    'norm_model_layers': norm_model_layers,
                    'sqrt_norm_model_layers': sqrt_norm_model_layers},
                'octaves_options': {
                    'num_octaves': num_octaves,
                    'octave_scale': octave_scale,
                    'unzoom_octave': unzoom_octave},
                'anti_saturation_options': anti_saturation_cfg,
                'extra_parameters': {
                    'outdir': run_dir,
                    'description': description}
            }
            # Save the run configuration
            gen_utils.save_config(ctx=ctx, run_dir=run_dir)

            # For each layer:
            for layer in layers:
                # Extract deep dream image
                dreamed_image = deep_dream(image, model, model_resolution, layers=[layer], channels=channels, seed=seed, normed=norm_model_layers,
                                           sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                           octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave, config=config)

                # Save the resulting dreamed image
                filename = f'layer-{layer}_dreamed_{os.path.basename(starting_image).split(".")[0]}.jpg'
                Image.fromarray(dreamed_image).save(os.path.join(run_dir, filename))

    else:
        if 'use_all' in layers:
            # Get all available layers
            layers = get_available_layers(max_resolution=model_resolution)
        else:
            # Parse the layers given by the user and leave only those available by the model
            available_layers = get_available_layers(max_resolution=model_resolution)
            layers = [layer for layer in layers if layer in available_layers]

        # Make the run dir in the specified output directory
        desc = get_run_desc(network_pkl, cfg, 'dream', f'layers_{layers_tag(layers)}')
        desc = f'{desc}-{description}' if len(description) != 0 else desc
        run_dir = gen_utils.make_run_dir(outdir, desc)

        if batch_size > 1 and {'b4_mbstd', 'b4_conv', 'fc', 'out'}.intersection(layers):
            print('Note: layers past b4_mbstd use minibatch statistics, so results with --batch-size > 1 '
                  'will differ slightly from running one seed at a time.')

        starting_images, used_seeds = [], []
        for chunk_start in range(0, len(seeds), batch_size):
            chunk_seeds = seeds[chunk_start:chunk_start + batch_size]

            # Get the images and image names for this batch of seeds
            images, image_names = [], []
            for seed in chunk_seeds:
                image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                                  starting_image=starting_image,
                                                  image_size=model_resolution,
                                                  convert_to_grayscale=convert_to_grayscale)
                images.append(image)
                image_names.append(starting_image)
                starting_image = None

            # Extract deep dream images, the whole batch in one pass (get_image already returns
            # model_resolution squares, so the initial crop/resize transform is a no-op)
            batch = torch.stack([preprocess(image) for image in images])
            dreamed_images = deep_dream(batch, model, model_resolution, layers=layers, channels=channels,
                                        seed=chunk_seeds[0] if len(chunk_seeds) == 1 else chunk_seeds,
                                        normed=norm_model_layers, sqrt_normed=sqrt_norm_model_layers,
                                        iterations=iterations, lr=learning_rate, octave_scale=octave_scale,
                                        num_octaves=num_octaves, unzoom_octave=unzoom_octave,
                                        ignore_initial_transform=True, config=config)
            if dreamed_images.ndim == 3:
                dreamed_images = dreamed_images[None]

            for image, image_name, seed, dreamed_image in zip(images, image_names, chunk_seeds, dreamed_images):
                # For logging later
                starting_images.append(image_name)
                used_seeds.append(seed)

                # Save the resulting image and initial image
                filename = f'dreamed_{os.path.basename(image_name)}'
                Image.fromarray(dreamed_image).save(os.path.join(run_dir, filename))
                image.save(os.path.join(run_dir, os.path.basename(image_name)))

        # Save the configuration used
        ctx.obj = {
            'network_pkl': network_pkl,
            'synthesis_options': {
                'seeds': used_seeds,
                'starting_image': starting_images,
                'class_idx': class_idx,
                'learning_rate': learning_rate,
                'iterations': iterations},
            'layer_options': {
                'layer': layers,
                'channels': channels,
                'norm_model_layers': norm_model_layers,
                'sqrt_norm_model_layers': sqrt_norm_model_layers},
            'octaves_options': {
                'octave_scale': octave_scale,
                'num_octaves': num_octaves,
                'unzoom_octave': unzoom_octave},
            'performance_options': {
                'batch_size': batch_size},
            'anti_saturation_options': anti_saturation_cfg,
            'extra_parameters': {
                'outdir': run_dir,
                'description': description}
        }
        # Save the run configuration
        gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


@main.command(name='dream-zoom',
              help='Zoom/rotate/translate after each Discriminator Dreaming iteration. A video will be saved.')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--random-image-noise', '-noise', 'image_noise', type=str, help='Starting image content: "random" noise, "perlin" noise, or any solid color understood by PIL (e.g., "white", "black", "#ff8800")', default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Comma-separated list of the layers of the Discriminator to use as the features. If "use_all", will use all available layers.', default='b16_conv0', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# Individual frame manipulation options
@click.option('--pixel-zoom', '-zoom', type=int, help='How many pixels to zoom per step (positive for zoom in, negative for zoom out, padded with black)', default=2, show_default=True)
@click.option('--rotation-deg', '-rot', type=float, help='Rotate image counter-clockwise per frame (padded with black)', default=0.0, show_default=True)
@click.option('--translate-x', '-tx', type=float, help='Translate the image in the horizontal axis per frame (from left to right, padded with black)', default=0.0, show_default=True)
@click.option('--translate-y', '-ty', type=float, help='Translate the image in the vertical axis per frame (from top to bottom, padded with black)', default=0.0, show_default=True)
# Video options
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
@click.option('--duration-sec', type=float, help='Duration length of the video', default=15.0, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Add flag to reverse the generated video')
@click.option('--include-starting-image', type=bool, help='Include the starting image in the final video', default=True, show_default=True)
# Anti-saturation options (jitter/flip default off for sequences to preserve temporal coherence)
@saturation_options()
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream_zoom(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seed: int,
        image_noise: Optional[str],
        starting_image: Optional[Union[str, os.PathLike]],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # For conditional models (not yet implemented)
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: List[int],
        norm_model_layers: Optional[bool],
        sqrt_norm_model_layers: Optional[bool],
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: Optional[bool],
        pixel_zoom: int,
        rotation_deg: float,
        translate_x: int,
        translate_y: int,
        fps: int,
        duration_sec: float,
        reverse_video: bool,
        include_starting_image: bool,
        anti_saturation: bool,
        soft_clip: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        grad_smooth: float,
        jitter: int,
        flip: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get the model resolution (for resizing the starting image if needed)
    model_resolution = D.img_resolution
    zoom_size = model_resolution - 2 * pixel_zoom

    layers = layers.split(',')
    if 'use_all' in layers:
        # Get all available layers
        layers = get_available_layers(max_resolution=model_resolution)
    else:
        # Parse the layers given by the user and leave only those available by the model
        available_layers = get_available_layers(max_resolution=model_resolution)
        layers = [layer for layer in layers if layer in available_layers]

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Anti-saturation configuration
    config = build_dream_config(anti_saturation, soft_clip, target_mean, target_std, grad_smooth, jitter, flip)

    # Get the image and image name
    image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                      starting_image=starting_image,
                                      image_size=model_resolution,
                                      convert_to_grayscale=convert_to_grayscale)

    # Make the run dir in the specified output directory
    desc = get_run_desc(network_pkl, cfg, 'dream-zoom', f'layers_{layers_tag(layers)}')
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seed': seed,
            'random_image_noise': image_noise,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations
        },
        'layer_options': {
            'layers': layers,
            'channels': channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers
        },
        'octaves_options': {
            'num_octaves': num_octaves,
            'octave_scale': octave_scale,
            'unzoom_octave': unzoom_octave
        },
        'frame_manipulation_options': {
            'pixel_zoom': pixel_zoom,
            'rotation_deg': rotation_deg,
            'translate_x': translate_x,
            'translate_y': translate_y,
        },
        'video_options': {
            'fps': fps,
            'duration_sec': duration_sec,
            'reverse_video': reverse_video,
            'include_starting_image': include_starting_image,
        },
        'anti_saturation_options': vars(config),
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    num_frames = int(np.rint(duration_sec * fps))  # Number of frames for the video
    n_digits = int(np.log10(num_frames)) + 1       # Number of digits for naming each frame

    # Save the starting image
    starting_image_name = f'dreamed_{0:0{n_digits}d}.jpg' if include_starting_image else 'starting_image.jpg'
    image.save(os.path.join(run_dir, starting_image_name))

    for idx, frame in enumerate(tqdm(range(num_frames), desc='Dreaming...', unit='frame')):
        # Zoom in after the first frame
        if idx > 0:
            image = crop_resize_rotate(image, crop_size=zoom_size, new_size=model_resolution,
                                       rotation_deg=rotation_deg, translate_x=translate_x, translate_y=translate_y)
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, seed=seed, normed=norm_model_layers,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations, channels=channels,
                                   lr=learning_rate, octave_scale=octave_scale, num_octaves=num_octaves,
                                   unzoom_octave=unzoom_octave, disable_inner_tqdm=True, config=config)

        # Save the resulting image and initial image
        filename = f'dreamed_{idx + 1:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image).save(os.path.join(run_dir, filename))

        # Now, the dreamed image is the starting image
        image = Image.fromarray(dreamed_image)

    # Save the final video
    save_video_ffmpeg(run_dir, f'dreamed_%0{n_digits}d.jpg', 'dream-zoom', fps=fps, reverse_video=reverse_video)


# ----------------------------------------------------------------------------

@main.command(name='channel-zoom', help='Dream zoom using only the specified channels in the selected layer')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--random-image-noise', '-noise', 'image_noise', type=str, help='Starting image content: "random" noise, "perlin" noise, or any solid color understood by PIL (e.g., "white", "black", "#ff8800")', default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layer', type=str, help='Layers of the Discriminator to use as the features.', default='b8_conv0', show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# Individual frame manipulation options
@click.option('--pixel-zoom', '-zoom', type=int, help='How many pixels to zoom per step (positive for zoom in, negative for zoom out, padded with black)', default=2, show_default=True)
@click.option('--rotation-deg', '-rot', type=float, help='Rotate image counter-clockwise per frame (padded with black)', default=0.0, show_default=True)
@click.option('--translate-x', '-tx', type=float, help='Translate the image in the horizontal axis per frame (from left to right, padded with black)', default=0.0, show_default=True)
@click.option('--translate-y', '-ty', type=float, help='Translate the image in the vertical axis per frame (from top to bottom, padded with black)', default=0.0, show_default=True)
# Video options
@click.option('--frames-per-channel', type=click.IntRange(min=1), help='Number of frames per channel', default=1, show_default=True)
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Add flag to reverse the generated video')
@click.option('--include-starting-image', type=bool, help='Include the starting image in the final video', default=True, show_default=True)
# Anti-saturation options (jitter/flip default off for sequences to preserve temporal coherence)
@saturation_options()
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def channel_zoom(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seed: int,
        image_noise: Optional[str],
        starting_image: Optional[Union[str, os.PathLike]],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # For conditional models (not yet implemented)
        learning_rate: float,
        iterations: int,
        layer: str,
        norm_model_layers: Optional[bool],
        sqrt_norm_model_layers: Optional[bool],
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: Optional[bool],
        pixel_zoom: int,
        rotation_deg: float,
        translate_x: int,
        translate_y: int,
        frames_per_channel: int,
        fps: int,
        reverse_video: bool,
        include_starting_image: bool,
        anti_saturation: bool,
        soft_clip: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        grad_smooth: float,
        jitter: int,
        flip: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    """Zoom in using all the channels of a network (or a specified layer)"""
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get the model resolution (for resizing the starting image if needed)
    model_resolution = D.img_resolution
    zoom_size = model_resolution - 2 * pixel_zoom

    if 'use_all' in layer:
        ctx.fail('Cannot use "use_all" with this command. Please specify the layers you want to use.')
    else:
        # Parse the layers given by the user and leave only those available by the model
        available_layers = get_available_layers(max_resolution=model_resolution)
        assert layer in available_layers, f'Layer {layer} not available. Available layers: {available_layers}'
        layers = [layer]

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Anti-saturation configuration
    config = build_dream_config(anti_saturation, soft_clip, target_mean, target_std, grad_smooth, jitter, flip)

    # Get the image and image name
    image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                      starting_image=starting_image,
                                      image_size=model_resolution,
                                      convert_to_grayscale=convert_to_grayscale)

    # Make the run dir in the specified output directory
    desc = get_run_desc(network_pkl, cfg, 'channel-zoom', f'layer_{layer}')
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Finally, let's get the number of channels in the selected layer
    channels_dict = {res: D.get_submodule(f'b{res}.conv0').out_channels for res in D.block_resolutions}
    channels_dict[4] = D.get_submodule('b4.conv').out_channels  # Last block has a different name
    # Get the dimension of the block from the selected layer (e.g., from 'b128_conv0' get '128')
    block_resolution = re.search(r'b(\d+)_', layer).group(1)
    total_channels = channels_dict[int(block_resolution)]
    # Make a list of all the channels, each repeated frames_per_channel
    channels = np.repeat(np.arange(total_channels), frames_per_channel)

    num_frames = int(np.rint(total_channels * frames_per_channel))  # Number of frames for the video
    n_digits = int(np.log10(num_frames)) + 1  # Number of digits for naming each frame

    # Save the starting image
    starting_image_name = f'dreamed_{0:0{n_digits}d}.jpg' if include_starting_image else 'starting_image.jpg'
    image.save(os.path.join(run_dir, starting_image_name))

    for idx, frame in enumerate(tqdm(range(num_frames), desc='Dreaming...', unit='frame')):
        # Zoom in after the first frame
        if idx > 0:
            image = crop_resize_rotate(image, crop_size=zoom_size, new_size=model_resolution,
                                       rotation_deg=rotation_deg, translate_x=translate_x, translate_y=translate_y)
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, seed=seed, normed=norm_model_layers,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations, channels=channels[idx:idx + 1],
                                   lr=learning_rate, octave_scale=octave_scale, num_octaves=num_octaves,
                                   unzoom_octave=unzoom_octave, disable_inner_tqdm=True, config=config)

        # Save the resulting image and initial image
        filename = f'dreamed_{idx + 1:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image).save(os.path.join(run_dir, filename))

        # Now, the dreamed image is the starting image
        image = Image.fromarray(dreamed_image)

    # Save the final video
    save_video_ffmpeg(run_dir, f'dreamed_%0{n_digits}d.jpg', 'channel-zoom', fps=fps, reverse_video=reverse_video)

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seed': seed,
            'random_image_noise': image_noise,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations
        },
        'layer_options': {
            'layer': layer,
            'channels': 'all',
            'total_channels': total_channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers
        },
        'octaves_options': {
            'num_octaves': num_octaves,
            'octave_scale': octave_scale,
            'unzoom_octave': unzoom_octave
        },
        'frame_manipulation_options': {
            'pixel_zoom': pixel_zoom,
            'rotation_deg': rotation_deg,
            'translate_x': translate_x,
            'translate_y': translate_y,
        },
        'video_options': {
            'fps': fps,
            'frames_per_channel': frames_per_channel,
            'reverse_video': reverse_video,
            'include_starting_image': include_starting_image,
        },
        'anti_saturation_options': vars(config),
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


@main.command(name='interp', help='Interpolate between two or more seeds')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='Random seeds to generate the Perlin noise from', required=True)
@click.option('--interp-type', '-interp', type=click.Choice(['linear', 'spherical']), help='Type of interpolation in Z or W', default='spherical', show_default=True)
@click.option('--smooth', is_flag=True, help='Add flag to smooth the interpolation between the seeds')
@click.option('--random-image-noise', '-noise', 'image_noise', type=str, help='Starting image content: "random" noise, "perlin" noise, or any solid color understood by PIL (e.g., "white", "black", "#ff8800")', default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Comma-separated list of the layers of the Discriminator to use as the features. If "use_all", will use all available layers.', default='b16_conv0', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# TODO: Individual frame manipulation options
# Video options
@click.option('--seed-sec', '-sec', type=float, help='Number of seconds between each seed transition', default=5.0, show_default=True)
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
# Anti-saturation options (jitter/flip default off for sequences to preserve temporal coherence)
@saturation_options()
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def random_interpolation(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seeds: List[int],
        interp_type: Optional[str],
        smooth: Optional[bool],
        image_noise: Optional[str],
        starting_image: Optional[Union[str, os.PathLike]],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # For conditional models (not yet implemented)
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: List[int],
        norm_model_layers: Optional[bool],
        sqrt_norm_model_layers: Optional[bool],
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: Optional[bool],
        seed_sec: float,
        fps: int,
        anti_saturation: bool,
        soft_clip: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        grad_smooth: float,
        jitter: int,
        flip: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    """
    Interpolate between random Perlin images and apply DeepDream.

    Note: For better temporal coherence, use the 'dream-video' command which
    generates true 3D Perlin noise instead of interpolating between 2D slices.
    """
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get model resolution
    model_resolution = D.img_resolution
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Anti-saturation configuration
    config = build_dream_config(anti_saturation, soft_clip, target_mean, target_std, grad_smooth, jitter, flip)

    layers = layers.split(',')
    # Get all available layers
    if 'use_all' in layers:
        layers = get_available_layers(max_resolution=model_resolution)
    else:
        # Parse the layers given by the user and leave only those available by the model
        available_layers = get_available_layers(max_resolution=model_resolution)
        layers = [layer for layer in layers if layer in available_layers]

    # Make the run dir in the specified output directory
    desc = get_run_desc(network_pkl, cfg, 'interp', f'layers_{layers_tag(layers)}')
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Number of steps to take between each random image
    n_steps = int(np.rint(seed_sec * fps))
    # Total number of frames
    num_frames = int(n_steps * (len(seeds) - 1))
    # Total video length in seconds
    duration_sec = num_frames / fps

    # Number of digits for naming purposes
    n_digits = int(np.log10(num_frames)) + 1

    # Create interpolation of noises
    random_images = []
    for seed in seeds:
        # Get the starting seed and image
        image, _ = get_image(seed=seed, image_noise=image_noise, starting_image=starting_image,
                             image_size=model_resolution, convert_to_grayscale=convert_to_grayscale)
        image = np.array(image) / 255.0
        random_images.append(image)
    random_images = np.stack(random_images)

    all_images = np.empty([0] + list(random_images.shape[1:]), dtype=np.float32)
    # Do interpolation
    for i in range(len(random_images) - 1):
        # Interpolate between each pair of images
        interp = gen_utils.interpolate(random_images[i], random_images[i + 1], n_steps, interp_type, smooth)
        # Append it to the list of all images
        all_images = np.append(all_images, interp, axis=0)

    # DeepDream expects a list of PIL.Image objects
    pil_images = []
    for idx in range(len(all_images)):
        im = (255 * all_images[idx]).astype(dtype=np.uint8)
        pil_images.append(Image.fromarray(im))

    for idx, image in enumerate(tqdm(pil_images, desc='Interpolating...', unit='frame', total=num_frames)):
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, channels=channels, seed=None,
                                   normed=norm_model_layers, disable_inner_tqdm=True, ignore_initial_transform=True,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                   octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave,
                                   config=config)

        # Save the resulting image and initial image
        filename = f'{image_noise}-interpolation_frame_{idx:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image).save(os.path.join(run_dir, filename))

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seeds': seeds,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations},
        'layer_options': {
            'layer': layers,
            'channels': channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers},
        'octaves_options': {
            'octave_scale': octave_scale,
            'num_octaves': num_octaves,
            'unzoom_octave': unzoom_octave},
        'anti_saturation_options': vars(config),
        'extra_parameters': {
            'outdir': run_dir,
            'description': description}
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Generate video
    save_video_ffmpeg(run_dir, f'{image_noise}-interpolation_frame_%0{n_digits}d.jpg', f'{image_noise}-interpolation', fps=fps)

# ----------------------------------------------------------------------------


def _slice_volume(volume: np.ndarray, axis: str) -> np.ndarray:
    """Reorder a (T, H, W, C) volume so the first dimension indexes slices perpendicular to `axis`"""
    if axis == 't':
        return volume                          # (T, H, W, C): xy planes
    elif axis == 'y':
        return volume.transpose(1, 0, 2, 3)    # (H, T, W, C): tx planes
    return volume.transpose(2, 0, 1, 3)        # (W, T, H, C): ty planes


def _dream_video_worker(rank: int, num_gpus: int, volume_path: str, run_dir: str, opts: dict) -> None:
    """
    Worker for (multi-GPU) dream-video generation. Worker `rank` loads the Discriminator on
    GPU `rank`, memory-maps the shared noise volume, and dreams the slice indices
    rank, rank + num_gpus, rank + 2*num_gpus, ... of every requested axis, `batch_size`
    slices per forward pass. Frames are named by their global slice index, so the parent
    can assemble the videos once all workers have finished.
    """
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    D = gen_utils.load_network('D', opts['network_pkl'], opts['cfg'], device)
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Memory-map the volume so each worker only reads the slices it needs
    volume = np.load(volume_path, mmap_mode='r')
    batch_size = opts['batch_size']

    for axis in opts['axes']:
        slices = _slice_volume(volume, axis)
        axis_dir = os.path.join(run_dir, f'axis_{axis}')
        n_digits = int(np.log10(len(slices))) + 1
        indices = list(range(rank, len(slices), num_gpus))

        pbar = tqdm(total=len(indices), desc=f'GPU {rank}: dreaming along the {axis}-axis',
                    unit='frame', position=rank)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            # Preprocess each slice and tile it up to a Discriminator-friendly size
            # (np.array copies the slice out of the read-only memory map)
            tensors = []
            for i in batch_indices:
                slice_tensor = preprocess(np.array(slices[i]))
                slice_tensor, (h, w) = pad_to_multiple(slice_tensor, opts['multiple'])
                tensors.append(slice_tensor)
            batch = torch.stack(tensors)

            dreamed = deep_dream(
                batch, model, opts['model_resolution'],
                layers=opts['layers'], channels=opts['channels'], seed=None,
                normed=opts['norm_model_layers'], sqrt_normed=opts['sqrt_norm_model_layers'],
                iterations=opts['iterations'], lr=opts['learning_rate'],
                octave_scale=opts['octave_scale'], num_octaves=opts['num_octaves'],
                unzoom_octave=opts['unzoom_octave'],
                disable_inner_tqdm=True,
                ignore_initial_transform=True,
                config=opts['dream_config']
            )
            if dreamed.ndim == 3:
                dreamed = dreamed[None]

            for j, i in enumerate(batch_indices):
                # Crop back to the slice size (made even, as yuv420p requires even dimensions)
                frame = dreamed[j][:h - h % 2, :w - w % 2]
                Image.fromarray(frame).save(os.path.join(axis_dir, f'frame_{i:0{n_digits}d}.jpg'))
            pbar.update(len(batch_indices))
        pbar.close()


def combine_axis_videos(video_paths: List[Union[str, os.PathLike]],
                        out_path: Union[str, os.PathLike],
                        height: int = 512,
                        crf: int = 20,
                        preset: str = 'veryfast') -> None:
    """Stack videos side by side (scaled to a common height, trimmed to the shortest one)"""
    n = len(video_paths)
    cmd = [_ffmpeg_bin(), '-y', '-loglevel', 'error']
    for path in video_paths:
        cmd += ['-i', str(path)]
    filters = ';'.join(f'[{i}:v]scale=-2:{height}[v{i}]' for i in range(n))
    filters += ';' + ''.join(f'[v{i}]' for i in range(n)) + f'hstack=inputs={n}:shortest=1[out]'
    cmd += ['-filter_complex', filters, '-map', '[out]',
            '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p', str(out_path)]
    subprocess.run(cmd, check=True)


@main.command(name='dream-video', help='DeepDream a 3D fractal Perlin noise volume, sliced along the t, y, and/or x axes')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the video')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Comma-separated list of discriminator layers to use', default='b16_conv0', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Channel indices to use (None = all)', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Divide features by number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Divide features by sqrt of number of elements')
# Octaves options (DeepDream image pyramid, not to be confused with the noise octaves)
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Unzoom octaves (needed for slices whose size the Discriminator cannot handle)', default=True, show_default=True)
# Noise volume options
@click.option('--axes', type=str, help='Comma-separated volume axes to slice along: "t" (xy planes), "y" (tx planes), "x" (ty planes)', default='t,y,x', show_default=True)
@click.option('--num-frames', type=click.IntRange(min=1), help='Size of the noise volume along the time axis', default=100, show_default=True)
@click.option('--img-size', type=click.IntRange(min=8), help='Size of the noise volume along the y and x axes (None = Discriminator resolution)', default=None)
@click.option('--noise-octaves', type=click.IntRange(min=1), help='Number of fractal octaves for the Perlin noise', default=6, show_default=True)
@click.option('--loop', is_flag=True, help='Make the noise periodic in time, so the t-axis video loops seamlessly')
# Performance options
@click.option('--gpus', type=click.IntRange(min=1), help='Number of GPUs to split the slices across (1 process per GPU)', default=1, show_default=True)
@click.option('--batch-size', type=click.IntRange(min=1), help='Number of slices to dream per forward pass on each GPU', default=1, show_default=True)
# Video options
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the output videos', default=25, show_default=True)
@click.option('--combine', type=bool, help='Stack the axis videos side by side into a single comparison video', default=True, show_default=True)
@click.option('--display-height', type=click.IntRange(min=64), help='Height of the combined comparison video', default=512, show_default=True)
# Anti-saturation options (jitter/flip default off for videos to preserve temporal coherence)
@saturation_options()
# Extra parameters
@click.option('--outdir', type=click.Path(file_okay=False), help='Output directory', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description for output directory', default='', show_default=True)
def discriminator_dream_video(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seed: int,
        convert_to_grayscale: bool,
        class_idx: Optional[int],
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: Optional[List[int]],
        norm_model_layers: bool,
        sqrt_norm_model_layers: bool,
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: bool,
        axes: str,
        num_frames: int,
        img_size: Optional[int],
        noise_octaves: int,
        loop: bool,
        gpus: int,
        batch_size: int,
        fps: int,
        combine: bool,
        display_height: int,
        anti_saturation: bool,
        soft_clip: bool,
        target_mean: Optional[float],
        target_std: Optional[float],
        grad_smooth: float,
        jitter: int,
        flip: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    """
    Generate DeepDream videos from a single 3D fractal Perlin noise volume.

    The volume has shape (num_frames, img_size, img_size) and is sliced along each
    requested axis: 't' yields num_frames xy-slices (the classic video), while 'y'
    and 'x' yield img_size slices of shape (num_frames, img_size) each. Every slice
    is DeepDreamed independently with the chosen Discriminator layers; the temporal
    coherence comes from the smoothness of the noise volume itself.

    Non-square slices are tiled up to a Discriminator-friendly size, dreamed, and
    cropped back, so the volume dimensions do not need to match the model resolution.

    Slices are independent of each other, so the work can be split across GPUs
    (--gpus N spawns one worker process per GPU) and batched (--batch-size B dreams
    B slices per forward pass); both are exactly equivalent to the sequential result,
    except for layers past b4_mbstd, whose minibatch statistics depend on the batch.
    """
    # Number of worker processes/GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if gpus > available_gpus:
            print(f'Requested {gpus} GPUs, but only {available_gpus} are visible; using {available_gpus}.')
        num_gpus = min(gpus, available_gpus)
    else:
        num_gpus = 1

    # Load the Discriminator on the CPU just to get its resolution; the workers load it on their own GPUs
    D = gen_utils.load_network('D', network_pkl, cfg, torch.device('cpu'))
    model_resolution = D.img_resolution
    del D

    # Parse layers
    layers = layers.split(',')
    if 'use_all' in layers:
        layers = get_available_layers(max_resolution=model_resolution)
    else:
        available_layers = get_available_layers(max_resolution=model_resolution)
        layers = [layer for layer in layers if layer in available_layers]
    assert len(layers) > 0, f'No valid layers given! Available layers: {get_available_layers(model_resolution)}'

    if batch_size > 1 and {'b4_mbstd', 'b4_conv', 'fc', 'out'}.intersection(layers):
        print('Note: layers past b4_mbstd use minibatch statistics, so results with --batch-size > 1 '
              'will differ slightly from running one slice at a time.')

    # Parse axes
    axes = [axis.strip() for axis in axes.split(',') if axis.strip()]
    assert all(axis in ('t', 'y', 'x') for axis in axes), f'Invalid axes "{axes}"; only "t", "y", and "x" are allowed'

    img_size = model_resolution if img_size is None else img_size

    # Both spatial dims of every slice must be divisible by this for the requested layers
    multiple = get_padding_multiple(layers, model_resolution)

    # Make output directory
    desc = get_run_desc(network_pkl, cfg, 'dream-video', f'axes_{"-".join(axes)}', f'layers_{layers_tag(layers)}')
    desc = f'{desc}-{description}' if description else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)
    for axis in axes:
        os.makedirs(os.path.join(run_dir, f'axis_{axis}'), exist_ok=True)

    # Anti-saturation configuration (picklable dataclass; passed to each worker)
    config = build_dream_config(anti_saturation, soft_clip, target_mean, target_std, grad_smooth, jitter, flip)

    # Generate the noise volume once and share it with the workers via a memory-mapped file
    print(f'Generating a ({num_frames}, {img_size}, {img_size}) 3D Perlin noise volume...')
    noise_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    volume = get_perlin_volume(seed, num_frames, img_size, convert_to_grayscale, noise_device, loop, noise_octaves)
    volume_path = os.path.join(run_dir, 'perlin_volume.npy')
    np.save(volume_path, volume)
    del volume
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    worker_opts = {
        'network_pkl': network_pkl,
        'cfg': cfg,
        'model_resolution': model_resolution,
        'layers': layers,
        'channels': channels,
        'norm_model_layers': norm_model_layers,
        'sqrt_norm_model_layers': sqrt_norm_model_layers,
        'iterations': iterations,
        'learning_rate': learning_rate,
        'octave_scale': octave_scale,
        'num_octaves': num_octaves,
        'unzoom_octave': unzoom_octave,
        'axes': axes,
        'multiple': multiple,
        'batch_size': batch_size,
        'dream_config': config,
    }

    if num_gpus > 1:
        print(f'Splitting the slices across {num_gpus} GPUs ({batch_size} slice(s) per forward pass each)...')
        torch.multiprocessing.spawn(_dream_video_worker, args=(num_gpus, volume_path, run_dir, worker_opts),
                                    nprocs=num_gpus, join=True)
    else:
        _dream_video_worker(0, 1, volume_path, run_dir, worker_opts)

    # Assemble one video per axis from the saved frames
    axis_videos = []
    for axis in axes:
        num_slices = num_frames if axis == 't' else img_size
        n_digits = int(np.log10(num_slices)) + 1
        axis_dir = os.path.join(run_dir, f'axis_{axis}')
        save_video_ffmpeg(axis_dir, f'frame_%0{n_digits}d.jpg', f'dream-video-{axis}_axis', fps=fps)
        axis_videos.append(os.path.join(axis_dir, f'dream-video-{axis}_axis.mp4'))

    # Stack the axis videos side by side for easy comparison
    if combine and len(axis_videos) > 1:
        print('Combining the axis videos side by side...')
        combine_axis_videos(axis_videos, os.path.join(run_dir, 'dream-video-combined.mp4'), height=display_height)

    # Save configuration
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seed': seed,
            'convert_to_grayscale': convert_to_grayscale,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations
        },
        'layer_options': {
            'layers': layers,
            'channels': channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers
        },
        'octaves_options': {
            'num_octaves': num_octaves,
            'octave_scale': octave_scale,
            'unzoom_octave': unzoom_octave
        },
        'video_options': {
            'axes': axes,
            'num_frames': num_frames,
            'img_size': img_size,
            'noise_octaves': noise_octaves,
            'loop': loop,
            'fps': fps,
            'combine': combine,
            'display_height': display_height
        },
        'performance_options': {
            'gpus': num_gpus,
            'batch_size': batch_size
        },
        'anti_saturation_options': vars(config),
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    print(f'Done! Results saved to {run_dir}')


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
