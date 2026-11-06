"""Ablation: plain vs transform-robust (EOT) protection vs a random-noise control.
Usage: python ablate.py --images test_images --n 8 --out ablation_v3.csv
Edit LPIPS = LPIPS(edit(post(original)), edit(post(variant))); higher = edit more destroyed.
jpeg50 is HELD OUT: robust training only sees JPEG quality 55-90.
"""
import argparse, glob, os, random

import lpips
import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from evaluate import ATTACKS, edit, lp
from protect import DEVICE, prep, protect, texture_mask, to_pil, to_tensor

PROMPT = "a person at a beach, photo"
ATK = ["none", "jpeg75", "jpeg50", "resize50", "blur"]
CONFIGS = [("eps8", 8, False), ("eps6_eot", 6, True), ("eps8_eot", 8, True), ("eps12_eot", 12, True)]


def noise_control(img, eps):
    x = to_tensor(img)
    g = torch.Generator(device=x.device).manual_seed(0)
    s = torch.sign(torch.randn(x.shape, generator=g, device=x.device))
    return to_pil(x + s * eps * texture_mask(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="ablation_v3.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*")))
    files = random.Random(0).sample(files, min(a.n, len(files)))  # same 8 images as v2 ablation

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    metric = lpips.LPIPS(net="alex").to(DEVICE)

    rows = []
    for f in files:
        orig = prep(Image.open(f))
        variants = {"noise_ctrl_eps8": noise_control(orig, 8 / 255)}
        for name, e, rob in CONFIGS:
            variants[name] = protect(orig, eps=e / 255, steps=a.steps, robust=rob)
        base = {k: edit(pipe, ATTACKS[k](orig), PROMPT) for k in ATK}
        for name, v in variants.items():
            p = psnr(np.array(orig), np.array(v))
            s = ssim(np.array(orig), np.array(v), channel_axis=2)
            for k in ATK:
                d = lp(metric, base[k], edit(pipe, ATTACKS[k](v), PROMPT))
                rows.append(dict(image=os.path.basename(f), variant=name, attack=k,
                                 psnr=p, ssim=s, edit_lpips=d))
        pd.DataFrame(rows).to_csv(a.out, index=False)  # save as we go
        print("done", f, flush=True)

    df = pd.DataFrame(rows)
    t = df.pivot_table(index="variant", columns="attack", values="edit_lpips")[ATK].round(3)
    t["psnr"] = df.groupby("variant")["psnr"].mean().round(1)
    print(t.to_markdown())


if __name__ == "__main__":
    main()
