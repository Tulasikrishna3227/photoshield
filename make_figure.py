"""Make a before/after figure for the README from ONE photo you own (or have permission to use).
Tiles: Original | Protected (robust) | SD edit of original | edit of PLAIN-protected | ...+JPEG q75
       | edit of ROBUST-protected | ...+JPEG q75
Uses a benign prompt and keeps the Stable Diffusion safety checker ON. Do not publish edits of
other people's photos without their permission.
Usage: python make_figure.py --image my_photo.jpg --out docs/demo.png [--prompt "..."] [--eps 8]
"""
import argparse
import io
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont

from protect import DEVICE, prep, protect


def jpeg(im, q=75):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return Image.open(b).convert("RGB")


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="docs/demo.png")
    ap.add_argument("--prompt", default="a person wearing a red winter jacket in a snowy street, photo")
    ap.add_argument("--eps", type=float, default=8)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()

    orig = prep(Image.open(a.image))
    plain = protect(orig, eps=a.eps / 255, steps=a.steps, robust=False)
    robust = protect(orig, eps=a.eps / 255, steps=a.steps, robust=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(  # safety checker left ON
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    def edit(im):
        g = torch.Generator(DEVICE).manual_seed(0)
        return pipe(a.prompt, image=im, strength=0.6, guidance_scale=7.5, num_inference_steps=30,
                    generator=g).images[0]

    tiles = [("Original", orig), ("Protected\n(robust, eps %g)" % a.eps, robust),
             ("SD edit of\noriginal", edit(orig)),
             ("Edit of\nplain-protected", edit(plain)), ("...after\nJPEG q75", edit(jpeg(plain))),
             ("Edit of\nrobust-protected", edit(robust)), ("...after\nJPEG q75", edit(jpeg(robust)))]
    th = 384
    w = int(orig.width * th / orig.height)
    bar = 64
    fig = Image.new("RGB", (w * len(tiles), th + bar), "white")
    d = ImageDraw.Draw(fig)
    f = font(20)
    for i, (label, im) in enumerate(tiles):
        fig.paste(im.resize((w, th)), (i * w, bar))
        d.multiline_text((i * w + 8, 6), label, fill="black", font=f)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.save(a.out)
    robust.save(os.path.splitext(a.out)[0] + "_protected.png")
    print("saved", a.out)


if __name__ == "__main__":
    main()
