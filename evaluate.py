"""Evaluate PhotoShield.

Usage: python evaluate.py --images ./test_images --out results.csv

For each image and each post-processing attack (none, jpeg75, jpeg50, resize50, blur) it measures:
  - PSNR / SSIM  original vs protected      (invisibility; higher = better)
  - LPIPS(edit(original), edit(protected))  (protection; HIGHER = edits more destroyed)
Edit model: SD1.5 img2img with fixed seed, so the only difference is the protection.
"""
import argparse
import glob
import io
import os

import lpips
import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from protect import DEVICE, prep, protect

PROMPTS = ["a person at a beach, photo", "a portrait in a studio, dramatic lighting"]


def jpeg75(im):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=75)
    return Image.open(b).convert("RGB")


def jpeg50(im):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=50)
    return Image.open(b).convert("RGB")


def resize50(im):
    w, h = im.size
    return im.resize((w // 2, h // 2), Image.LANCZOS).resize((w, h), Image.LANCZOS)


ATTACKS = {
    "none": lambda im: im,
    "jpeg75": jpeg75,
    "jpeg50": jpeg50,  # held out: robust training uses quality 55-90 only
    "resize50": resize50,
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
}


def edit(pipe, im, prompt, seed=0):
    g = torch.Generator(DEVICE).manual_seed(seed)
    return pipe(prompt=prompt, image=im, strength=0.6, guidance_scale=7.5,
                num_inference_steps=30, generator=g).images[0]


def lp(metric, a, b):
    t = lambda im: (torch.from_numpy(np.array(im)).float() / 127.5 - 1).permute(2, 0, 1)[None].to(DEVICE)
    with torch.no_grad():
        return metric(t(a), t(b)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--eps", type=float, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--robust", action="store_true", help="transform-robust (EOT) protection")
    a = ap.parse_args()

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    metric = lpips.LPIPS(net="alex").to(DEVICE)

    rows = []
    files = sorted(glob.glob(os.path.join(a.images, "*")))
    for f in files:
        orig = prep(Image.open(f))
        prot = protect(orig, eps=a.eps / 255, steps=a.steps, robust=a.robust)
        s = ssim(np.array(orig), np.array(prot), channel_axis=2)
        p = psnr(np.array(orig), np.array(prot))
        for name, atk in ATTACKS.items():
            o, q = atk(orig), atk(prot)
            for pr in PROMPTS:
                d = lp(metric, edit(pipe, o, pr), edit(pipe, q, pr))
                rows.append(dict(image=os.path.basename(f), attack=name, prompt=pr,
                                 psnr=p, ssim=s, edit_lpips=d))
        print("done", f)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby("attack")[["psnr", "ssim", "edit_lpips"]].mean().round(3).to_markdown())


if __name__ == "__main__":
    main()
