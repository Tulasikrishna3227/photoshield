"""Make a before/after figure for the README from ONE photo you own.
Columns: original | protected | SD edit of original | SD edit of protected | same after JPEG q75.
Usage: python make_figure.py --image my_photo.jpg --out docs/demo.png [--prompt "..."] [--eps 8]
"""
import argparse
import io
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw

from protect import DEVICE, prep, protect


def jpeg(im, q=75):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return Image.open(b).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="docs/demo.png")
    ap.add_argument("--prompt", default="a person at a beach, photo")
    ap.add_argument("--eps", type=float, default=8)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()

    orig = prep(Image.open(a.image))
    prot = protect(orig, eps=a.eps / 255, steps=a.steps, robust=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    def edit(im):
        g = torch.Generator(DEVICE).manual_seed(0)
        return pipe(a.prompt, image=im, strength=0.6, guidance_scale=7.5, num_inference_steps=30,
                    generator=g).images[0]

    tiles = [("Original", orig), ("Protected", prot), ("Edit of original", edit(orig)),
             ("Edit of protected", edit(prot)), ("Edit of protected\n(after JPEG q75)", edit(jpeg(prot)))]
    w, h = orig.size
    bar = 44
    fig = Image.new("RGB", (w * len(tiles), h + bar), "white")
    d = ImageDraw.Draw(fig)
    for i, (label, im) in enumerate(tiles):
        fig.paste(im.resize((w, h)), (i * w, bar))
        d.text((i * w + 8, 6), label, fill="black")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.save(a.out)
    prot.save(os.path.splitext(a.out)[0] + "_protected.png")
    print("saved", a.out)


if __name__ == "__main__":
    main()
