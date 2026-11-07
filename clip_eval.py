"""Prompt-adherence evaluation (PhotoGuard-style). Saves metrics only, no images.

For each image: CLIP(prompt, edit) for the edit of the original, then for edits of each protected
variant, under no post-processing and JPEG q75. A destroyed edit stops matching the prompt, so
drop = CLIP(edit of original) - CLIP(edit of variant) should be large and clearly above the
random-noise control's drop. Thresholds are provisional until we see the control's spread.
Usage: python clip_eval.py --images test_images --n 16 --out clip_eval.csv
"""
import argparse, glob, os, random

import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from transformers import CLIPModel, CLIPProcessor
import numpy as np

from ablate import noise_control
from evaluate import ATTACKS, edit
from protect import DEVICE, prep, protect

PROMPT = "a person wearing a red winter jacket in a snowy street, photo"
ATK = ["none", "jpeg75"]


class Clip:
    def __init__(self):
        name = "openai/clip-vit-base-patch32"
        self.m = CLIPModel.from_pretrained(name).to(DEVICE).eval()
        self.p = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def score(self, im, text):
        x = self.p(text=[text], images=im, return_tensors="pt", padding=True).to(DEVICE)
        i = self.m.get_image_features(pixel_values=x["pixel_values"])
        t = self.m.get_text_features(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        return torch.nn.functional.cosine_similarity(i, t).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="clip_eval.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*")))
    files = random.Random(0).sample(files, min(a.n, len(files)))

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    clip = Clip()

    rows = []
    for idx, f in enumerate(files):
        orig = prep(Image.open(f))
        variants = {"noise_ctrl_eps8": noise_control(orig, 8 / 255),
                    "plain_eps8": protect(orig, eps=8 / 255),
                    "robust_eps8": protect(orig, eps=8 / 255, robust=True),
                    "plain_eps16_s200": protect(orig, eps=16 / 255, steps=200)}
        base = {k: clip.score(edit(pipe, ATTACKS[k](orig), PROMPT), PROMPT) for k in ATK}
        for name, v in variants.items():
            p = psnr(np.array(orig), np.array(v))
            for k in ATK:
                s = clip.score(edit(pipe, ATTACKS[k](v), PROMPT), PROMPT)
                rows.append(dict(image=f"img{idx + 1:02d}", variant=name, attack=k, psnr=p,
                                 clip_orig=base[k], clip_var=s, drop=base[k] - s))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("done", idx + 1, "/", len(files), flush=True)

    df = pd.DataFrame(rows)
    t = df.pivot_table(index="variant", columns="attack", values="drop").round(3)
    t["psnr"] = df.groupby("variant").psnr.mean().round(1)
    print("mean CLIP prompt-adherence drop (higher = more broken):")
    print(t.to_markdown())
    print("\nimages with drop > 0.05, no post-processing:")
    print(df[df.attack == "none"].assign(hit=lambda d: d["drop"] > 0.05)
          .groupby("variant").hit.sum().to_string())


if __name__ == "__main__":
    main()
