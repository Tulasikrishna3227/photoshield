"""Low-strength img2img evaluation: does the protected image's edit still stay close to the source?

shift = LPIPS(edit(variant), original) - LPIPS(edit(original), original). At low strength an edit
of an unprotected photo stays close to the photo; a working protection makes the edit drift or
break, so shift should be clearly positive and clearly above the random-noise control.
Metrics only, no images saved. Usage: python strength_eval.py --images test_images --n 16
"""
import argparse, glob, os, random

import lpips
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from ablate import noise_control
from evaluate import lp
from protect import DEVICE, prep, protect

PROMPT = "a person wearing a red jacket, studio photo"
STRENGTHS = [0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="strength_eval.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*")))
    files = random.Random(0).sample(files, min(a.n, len(files)))

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    metric = lpips.LPIPS(net="alex").to(DEVICE)

    def edit(im, s):
        g = torch.Generator(DEVICE).manual_seed(0)
        return pipe(PROMPT, image=im, strength=s, guidance_scale=7.5, num_inference_steps=30,
                    generator=g).images[0]

    rows = []
    for idx, f in enumerate(files):
        orig = prep(Image.open(f))
        variants = {"noise_ctrl_eps8": noise_control(orig, 8 / 255),
                    "plain_eps8": protect(orig, eps=8 / 255),
                    "robust_eps8": protect(orig, eps=8 / 255, robust=True),
                    "plain_eps16_s200": protect(orig, eps=16 / 255, steps=200)}
        for s in STRENGTHS:
            base = lp(metric, edit(orig, s), orig)
            for name, v in variants.items():
                d = lp(metric, edit(v, s), orig)
                rows.append(dict(image=f"img{idx + 1:02d}", variant=name, strength=s,
                                 lpips_to_src=d, base=base, shift=d - base))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("done", idx + 1, "/", len(files), flush=True)

    df = pd.DataFrame(rows)
    print("mean shift (higher = edit drifts / breaks more):")
    print(df.pivot_table(index="variant", columns="strength", values="shift").round(3).to_markdown())
    print("\nimages with shift > 0.10:")
    print(df.assign(hit=lambda d: d["shift"] > 0.10).pivot_table(
        index="variant", columns="strength", values="hit", aggfunc="sum").to_markdown())


if __name__ == "__main__":
    main()
