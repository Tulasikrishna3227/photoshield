"""PhotoShield core: masked PGD encoder attack on the Stable Diffusion VAE (PhotoGuard-style),
with optional transform-robust training (EOT over differentiable JPEG / resize / blur).

Find a small perturbation delta (|delta| <= eps * texture_mask) so the VAE encoder maps
(image + delta) to a garbage target latent. SD-based editors start from that latent, so their
output gets destroyed. `robust=True` optimises through random JPEG/resize/blur so the
protection survives them better.
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vae = None


# ---------------------------------------------------------------- image utils
def get_vae():
    global _vae
    if _vae is None:
        _vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
        for p in _vae.parameters():
            p.requires_grad_(False)
    return _vae


def prep(img, max_side=512):
    """RGB, longest side <= max_side, dims multiple of 8 (VAE and JPEG-block requirement)."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    w, h = max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)
    return img.resize((w, h), Image.LANCZOS)


def to_tensor(img):
    return torch.from_numpy(np.array(img)).float().div(255).permute(2, 0, 1)[None].to(DEVICE)


def to_pil(t):
    a = t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((a * 255).round().astype(np.uint8))


def _target(shape, tile=24):
    """High-frequency checkerboard: a latent far from any natural photo."""
    _, _, h, w = shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    board = (((yy // tile) + (xx // tile)) % 2).float()
    return board[None, None].repeat(1, 3, 1, 1).to(DEVICE)


def texture_mask(x, lo=0.1):
    """1 on textured areas, `lo` on flat areas (wall, skin): hides noise where eyes notice it."""
    g = x.mean(1, keepdim=True)
    gx = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
    m = F.avg_pool2d(gx + gy, 9, 1, 4)
    m = (m / torch.quantile(m.flatten(), 0.95).clamp_min(1e-6)).clamp(0, 1)
    return lo + (1 - lo) * m


# ------------------------------------------------- differentiable transforms (EOT)
_Q = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
                  dtype=torch.float32)


def _dct_matrix(n=8):
    k = torch.arange(n).float()
    d = torch.cos((2 * k[None, :] + 1) * k[:, None] * math.pi / (2 * n)) * math.sqrt(2 / n)
    d[0] /= math.sqrt(2)
    return d


def _round_ste(x):
    return x + (x.round() - x).detach()  # straight-through rounding


def jpeg_approx(x, quality):
    """Differentiable JPEG approximation on RGB channels (8x8 DCT + quantisation, STE rounding).
    x: (B,3,H,W) in [0,1], H and W multiples of 8. Not real JPEG (no chroma subsampling), but
    within ~0.5 grey levels of PIL's JPEG on luminance in our check."""
    b_, c, h, w = x.shape
    d = _dct_matrix().to(x.device)
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = torch.clamp(torch.floor((_Q.to(x.device) * s + 50) / 100), 1, 255)
    blocks = (x * 255 - 128).reshape(b_, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)
    coef = _round_ste((d @ blocks @ d.T) / q) * q
    rec = (d.T @ coef @ d).permute(0, 1, 2, 4, 3, 5).reshape(b_, c, h, w)
    return ((rec + 128) / 255).clamp(0, 1)


def rand_resize(x, s):
    h, w = x.shape[-2:]
    y = F.interpolate(x, size=(max(8, int(h * s)), max(8, int(w * s))),
                      mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def gaussian_blur(x, sigma, k=5):
    ax = torch.arange(k, device=x.device).float() - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kern = (g[:, None] * g[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kern, groups=x.shape[1])


def random_transform(x, rng):
    r = rng.random()
    if r < 0.2:
        return x
    if r < 0.5:
        return jpeg_approx(x, rng.uniform(55, 90))
    if r < 0.75:
        return rand_resize(x, rng.uniform(0.4, 0.9))
    return gaussian_blur(x, rng.uniform(0.5, 1.5))


# ------------------------------------------------------------------- protection
def protect(img, eps=8 / 255, steps=100, robust=False, mask=True, lo=0.1, seed=0, progress=None):
    """Return a protected copy of `img` (resized by prep). Save as PNG, not JPEG."""
    vae = get_vae()
    rng = random.Random(seed)
    x = to_tensor(prep(img))
    bound = eps * (texture_mask(x, lo) if mask else torch.ones_like(x[:, :1]))
    with torch.no_grad():
        z_t = vae.encode(_target(x.shape) * 2 - 1).latent_dist.mean
    alpha = eps / 10
    delta = torch.zeros_like(x)
    for i in range(steps):
        delta.requires_grad_(True)
        xa = (x + delta).clamp(0, 1)
        if robust:
            xa = random_transform(xa, rng)
        z = vae.encode(xa * 2 - 1).latent_dist.mean
        (grad,) = torch.autograd.grad(F.mse_loss(z, z_t), delta)
        delta = delta - alpha * grad.sign()
        delta = torch.max(torch.min(delta, bound), -bound)
        delta = ((x + delta).clamp(0, 1) - x).detach()
        if progress:
            progress((i + 1) / steps)
    return to_pil(x + delta)


if __name__ == "__main__":
    import sys
    protect(Image.open(sys.argv[1]), robust="--robust" in sys.argv).save(sys.argv[2], format="PNG")
    print("saved", sys.argv[2])
