"""
Fractal Perlin Noise generators for 2D images and 3D video sequences.

Based on the pyperlin library, extended to support 3D (time-based) noise for smooth video generation.
"""
import torch
import numpy as np
from typing import List, Tuple

tau = 6.28318530718


class FractalPerlin2D(object):
    """
    2D Fractal Perlin Noise generator.

    Args:
        shape: Tuple of (channels, height, width) or (height, width)
        resolutions: List of (height_res, width_res) tuples for each octave
        factors: List of amplitude factors for each octave
        generator: PyTorch random generator (for reproducibility)
    """

    def __init__(self, shape: Tuple[int, ...], resolutions: List[Tuple[int, int]],
                 factors: List[float], generator=torch.random.default_generator):
        shape = shape if len(shape) == 3 else (None,) + shape
        self.shape = shape
        self.factors = factors
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions
        self.grid_shapes = [(shape[1] // res[0], shape[2] // res[1]) for res in resolutions]

        # Precomputed tensors
        self.linxs = [torch.linspace(0, 1, gs[1], device=self.device) for gs in self.grid_shapes]
        self.linys = [torch.linspace(0, 1, gs[0], device=self.device) for gs in self.grid_shapes]
        self.tl_masks = [self.fade(lx)[None, :] * self.fade(ly)[:, None] for lx, ly in zip(self.linxs, self.linys)]
        self.tr_masks = [torch.flip(tl_mask, dims=[1]) for tl_mask in self.tl_masks]
        self.bl_masks = [torch.flip(tl_mask, dims=[0]) for tl_mask in self.tl_masks]
        self.br_masks = [torch.flip(tl_mask, dims=[0, 1]) for tl_mask in self.tl_masks]

    def fade(self, t: torch.Tensor) -> torch.Tensor:
        """Smoothstep fade function: 6t^5 - 15t^4 + 10t^3"""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise(self, octave: int, batch_size: int) -> torch.Tensor:
        """Generate Perlin noise for a specific octave."""
        res = self.resolutions[octave]
        angles = torch.zeros((batch_size, res[0] + 2, res[1] + 2), device=self.device)
        angles.uniform_(0, tau, generator=self.generator)
        rx = torch.cos(angles)[:, :, :, None] * self.linxs[octave]
        ry = torch.sin(angles)[:, :, :, None] * self.linys[octave]
        prx, pry = rx[:, :, :, None, :], ry[:, :, :, :, None]
        nrx, nry = -torch.flip(prx, dims=[4]), -torch.flip(pry, dims=[3])
        br = prx[:, :-1, :-1] + pry[:, :-1, :-1]
        bl = nrx[:, :-1, 1:] + pry[:, :-1, 1:]
        tr = prx[:, 1:, :-1] + nry[:, 1:, :-1]
        tl = nrx[:, 1:, 1:] + nry[:, 1:, 1:]

        grid_shape = self.grid_shapes[octave]
        grids = (self.br_masks[octave] * br + self.bl_masks[octave] * bl +
                 self.tr_masks[octave] * tr + self.tl_masks[octave] * tl)
        noise = grids.permute(0, 1, 3, 2, 4).reshape(
            (batch_size, self.shape[1] + grid_shape[0], self.shape[2] + grid_shape[1]))

        A = torch.randint(0, grid_shape[0], (batch_size,), device=self.device, generator=self.generator)
        B = torch.randint(0, grid_shape[1], (batch_size,), device=self.device, generator=self.generator)
        noise = torch.stack([noise[n, a:a - grid_shape[0], b:b - grid_shape[1]]
                           for n, (a, b) in enumerate(zip(A, B))])
        return noise

    def __call__(self, batch_size: int = None) -> torch.Tensor:
        """Generate 2D fractal Perlin noise."""
        batch_size = self.shape[0] if batch_size is None else batch_size
        shape = (batch_size,) + self.shape[1:]
        noise = torch.zeros(shape, device=self.device)
        for octave, factor in enumerate(self.factors):
            noise += factor * self.perlin_noise(octave, batch_size=batch_size)
        return noise


class FractalPerlin3D(object):
    """
    3D Fractal Perlin Noise generator for smooth video sequences.

    Proper lattice-based Perlin noise: each octave places random unit gradient
    vectors on a coarse (t, y, x) lattice and interpolates their dot products
    with the local offset vectors using the smoothstep fade. Summing octaves
    with decreasing amplitudes gives the fractal (fBm) result.

    The resulting volume is isotropic, so it can be sliced along any of the
    three axes (xy, ty, tx planes) and still look like 2D Perlin noise.

    Args:
        shape: Tuple of (channels, height, width); each channel gets an independent noise field
        resolutions: List of (t_cells, y_cells, x_cells) lattice cells per octave
        factors: List of amplitude factors for each octave
        num_frames: Total number of frames to generate
        generator: PyTorch random generator (for reproducibility)
        loop: If True, the noise is periodic along the time axis (seamless loop)
    """

    def __init__(self, shape: Tuple[int, int, int], resolutions: List[Tuple[int, int, int]],
                 factors: List[float], num_frames: int,
                 generator=torch.random.default_generator, loop: bool = True):
        shape = shape if len(shape) == 3 else (1,) + tuple(shape)
        self.channels, self.height, self.width = shape
        self.factors = factors
        self.num_frames = num_frames
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions  # [(t_cells, y_cells, x_cells), ...]
        self.loop = loop

    @staticmethod
    def fade(t: torch.Tensor) -> torch.Tensor:
        """Smoothstep fade function: 6t^5 - 15t^4 + 10t^3"""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise_3d(self, octave: int) -> torch.Tensor:
        """Generate one octave of 3D Perlin noise; returns a (num_frames, H, W) tensor."""
        T, H, W = self.num_frames, self.height, self.width
        ct, cy, cx = self.resolutions[octave]
        device = self.device

        # Random unit gradient vectors on the (ct+1, cy+1, cx+1) lattice (uniform on the sphere)
        g_shape = (ct + 1, cy + 1, cx + 1)
        theta = torch.rand(g_shape, generator=self.generator, device=device) * tau
        gt = torch.rand(g_shape, generator=self.generator, device=device) * 2 - 1
        s = torch.sqrt(torch.clamp(1 - gt**2, min=0.0))
        grads = torch.stack((s * torch.cos(theta), s * torch.sin(theta), gt), dim=-1)  # (..., [gx, gy, gt])
        if self.loop:
            grads[-1] = grads[0]  # Periodic along time => seamless loop

        # Voxel coordinates in lattice space: cell index + fractional position
        tc = torch.arange(T, device=device, dtype=torch.float32) * (ct / T)
        yc = torch.arange(H, device=device, dtype=torch.float32) * (cy / H)
        xc = torch.arange(W, device=device, dtype=torch.float32) * (cx / W)
        y0 = yc.long().clamp_(max=cy - 1)
        x0 = xc.long().clamp_(max=cx - 1)
        fy = (yc - y0).view(1, H, 1)
        fx = (xc - x0).view(1, 1, W)
        wy, wx = self.fade(fy), self.fade(fx)

        noise = torch.empty((T, H, W), device=device)
        # Chunk along time to bound the memory of the (chunk, H, W, 3) gradient gathers
        chunk = max(1, 2**22 // (H * W))
        for start in range(0, T, chunk):
            ts = tc[start:start + chunk]
            t0 = ts.long().clamp_(max=ct - 1)
            ft = (ts - t0).view(-1, 1, 1)
            wt = self.fade(ft)
            acc = torch.zeros((len(ts), H, W), device=device)
            # Trilinear interpolation of the 8 corner gradient dot products
            for dt in (0, 1):
                w_t = wt if dt else 1 - wt
                idx_t = (t0 + dt).view(-1, 1, 1)
                for dy in (0, 1):
                    w_ty = w_t * (wy if dy else 1 - wy)
                    idx_y = (y0 + dy).view(1, -1, 1)
                    for dx in (0, 1):
                        weight = w_ty * (wx if dx else 1 - wx)
                        g = grads[idx_t, idx_y, (x0 + dx).view(1, 1, -1)]  # (chunk, H, W, 3)
                        dot = g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy) + g[..., 2] * (ft - dt)
                        acc += weight * dot
            noise[start:start + chunk] = acc
        return noise

    def __call__(self) -> torch.Tensor:
        """
        Generate 3D fractal Perlin noise.

        Returns:
            Tensor of shape (num_frames, C, H, W) - a video sequence,
            with an independent noise field per channel
        """
        channels = []
        for _ in range(self.channels):
            noise = torch.zeros((self.num_frames, self.height, self.width), device=self.device)
            for octave, factor in enumerate(self.factors):
                noise += factor * self.perlin_noise_3d(octave)
            channels.append(noise)
        return torch.stack(channels, dim=1)  # (num_frames, C, H, W)


# Convenience functions
def get_2d_perlin(shape: Tuple[int, int, int], seed: int = 0,
                  device: str = 'cuda', lacunarity: float = 2.0,
                  persistence: float = 0.5, octaves: int = 6) -> torch.Tensor:
    """
    Generate 2D fractal Perlin noise with default parameters.

    Args:
        shape: (C, H, W) tuple; each channel gets an independent noise field
        seed: Random seed
        device: Device to generate on ('cuda' or 'cpu')
        lacunarity: Frequency multiplier between octaves (currently fixed at 2.0,
                    as FractalPerlin2D requires H and W to be divisible by the cell counts)
        persistence: Amplitude multiplier between octaves (default 0.5)
        octaves: Number of octaves to sum (clamped so the finest lattice fits the image)

    Returns:
        Tensor of shape (C, H, W) with values approximately in [-1, 1]
    """
    # The finest lattice (2**octaves cells) must divide the image size
    octaves = min(octaves, int(np.log2(min(shape[1], shape[2]))))
    resolutions = [(2**i, 2**i) for i in range(1, octaves + 1)]
    factors = [persistence**i for i in range(octaves)]
    g = torch.Generator(device=device).manual_seed(seed)
    noise = FractalPerlin2D(shape, resolutions, factors, generator=g)()  # batch dim = channels
    return noise


def get_3d_perlin(shape: Tuple[int, int, int], num_frames: int, seed: int = 0,
                  device: str = 'cuda', lacunarity: float = 2.0,
                  persistence: float = 0.5, octaves: int = 6,
                  loop: bool = True, isotropic: bool = True) -> torch.Tensor:
    """
    Generate 3D fractal Perlin noise for video sequences.

    Args:
        shape: (C, H, W) tuple for spatial dimensions; each channel is an independent field
        num_frames: Number of frames to generate
        seed: Random seed
        device: Device to generate on ('cuda' or 'cpu')
        lacunarity: Frequency multiplier between octaves (default 2.0)
        persistence: Amplitude multiplier between octaves (default 0.5)
        octaves: Number of octaves to sum
        loop: If True, the video will loop seamlessly (periodic along time)
        isotropic: If True, scale the time-axis cell count by num_frames / max(H, W),
                   so a lattice cell spans the same "world distance" along every axis;
                   this makes slices along t, y, and x statistically equivalent

    Returns:
        Tensor of shape (num_frames, C, H, W) with values approximately in [-1, 1]
    """
    T = num_frames
    S = max(shape[1], shape[2])
    resolutions = []
    for i in range(octaves):
        cells = 2 * lacunarity**i
        cy = max(1, min(round(cells), shape[1]))
        cx = max(1, min(round(cells), shape[2]))
        ct = max(1, round(cells * T / S)) if isotropic else max(1, min(round(cells), T))
        resolutions.append((ct, cy, cx))
    factors = [persistence**i for i in range(octaves)]
    g = torch.Generator(device=device).manual_seed(seed)
    noise = FractalPerlin3D(shape, resolutions, factors, num_frames, generator=g, loop=loop)()
    return noise
