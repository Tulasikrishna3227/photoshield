# PhotoShield: adversarial protection against AI photo editing

Adds an invisible, texture-masked perturbation to a photo so Stable Diffusion based editors
(img2img, inpainting) produce destroyed outputs. PGD encoder attack on the SD VAE
(PhotoGuard-style), plus an optional transform-robust mode that optimises through
differentiable JPEG / resize / blur.

## Run
    pip install -r requirements.txt
    python app.py                          # web demo (Gradio)
    python protect.py in.jpg out.png       # CLI; add --robust for robust mode. Always save PNG.
    python evaluate.py --images ./test_images --eps 8 --out results.csv
    python ablate.py   --images ./test_images --n 8 --out ablation_v3.csv
    python make_figure.py --image my_photo.jpg --out docs/demo.png   # before/after figure (use your own photo)

## Method
1. Encode a target (high-frequency checkerboard) with the SD VAE to get a "garbage" latent.
2. PGD (100 steps) finds delta so VAE(x + delta) matches that latent, with |delta| <= eps * mask.
3. The mask is high on textured regions (hair, fabric) and low on flat regions (walls, skin),
   which removed the visible blotches a uniform eps produced.
4. Robust mode: each PGD step passes x + delta through a random differentiable transform
   (JPEG approx via 8x8 DCT + straight-through rounding, resize, Gaussian blur).

## Results
Metric: LPIPS between the SD1.5 img2img edit of the original and of the protected image
(same prompt and seed). Higher = edit more destroyed. PSNR/SSIM measure invisibility.

### Baseline, eps 6/255, masked, plain PGD (16 images, 2 prompts)
| Post-processing | PSNR | SSIM | Edit LPIPS |
|---|---|---|---|
| none | 40.3 | 0.985 | 0.422 |
| JPEG q75 | 40.3 | 0.985 | 0.224 |
| blur | 40.3 | 0.985 | 0.203 |
| resize 50% | 40.3 | 0.985 | 0.186 |

### Ablation v2: eps and Gaussian-noise training vs random-noise control (8 images, 1 prompt)
| Variant | none | JPEG q75 | resize 50% | PSNR |
|---|---|---|---|---|
| random-noise control (eps 6) | 0.209 | 0.134 | 0.079 | 39.6 |
| eps 4 | 0.340 | 0.181 | 0.112 | 43.6 |
| eps 6 | 0.425 | 0.251 | 0.212 | 40.4 |
| eps 8 | 0.442 | 0.277 | 0.207 | 38.1 |
| eps 12 | 0.503 | 0.308 | 0.258 | 34.9 |
| eps 6, Gaussian-noise training | 0.352 | 0.217 | 0.193 | 40.9 |
| eps 8, Gaussian-noise training | 0.392 | 0.279 | 0.241 | 38.5 |
| eps 12, Gaussian-noise training | 0.427 | 0.321 | 0.251 | 35.1 |

Findings: the perturbation is far stronger than random noise of equal magnitude, but a large
part of the effect is lost after JPEG/resize. Training through Gaussian noise did not help;
v3 below fixes this with transform-aware training.

### Ablation v3: transform-robust (EOT) training (8 images, 1 prompt)
JPEG q50 is held out of training (robust training samples JPEG quality 55-90, so q50 is only
slightly outside that range).

| Variant | none | JPEG q75 | JPEG q50 | resize 50% | blur | PSNR |
|---|---|---|---|---|---|---|
| random-noise control (eps 8) | 0.255 | 0.156 | 0.170 | 0.097 | 0.165 | 37.2 |
| eps 8, plain PGD | 0.445 | 0.279 | 0.249 | 0.224 | 0.271 | 38.1 |
| eps 6, robust | 0.378 | 0.310 | 0.269 | 0.282 | 0.307 | 40.7 |
| eps 8, robust | 0.406 | 0.336 | 0.304 | 0.310 | 0.336 | 38.3 |
| eps 12, robust | 0.462 | 0.381 | 0.353 | 0.384 | 0.390 | 34.9 |

Findings:
- At eps 8, robust training raises the score after post-processing by about 20% (JPEG q75),
  22% (JPEG q50, held out), 38% (resize) and 24% (blur), at a cost of about 9% on unprocessed
  images, with PSNR unchanged.
- Robust eps 6 beats plain eps 8 on average on every post-processed column while being less
  visible (40.7 vs 38.1 dB); per image it wins 4/8 (JPEG q75) to 7/8 (blur), so that one is
  a weaker claim than the eps 8 comparison.
- Per-image paired comparison, robust eps 8 vs plain eps 8: higher after post-processing in
  28 of 32 image/attack pairs (resize 8/8, JPEG q50 7/8, blur 7/8, JPEG q75 6/8); lower on
  unprocessed images in 7 of 8. The JPEG q75 gain is within noise at n=8 (+0.057, s.e. 0.040);
  the resize, blur and JPEG q50 gains are about 2.5 to 4 standard errors.
- Every protected variant stays well above the random-noise control.
- Still only partial protection: scores after JPEG/resize are below the unprocessed scores.

## Limitations
- Small evaluation: 16 images (15 web-sourced portraits not redistributed here, 1 personal
  photo), 1-2 prompts, one seed. No confidence intervals.
- LPIPS between edits measures how different the edits are, not how "broken" they look;
  include visual examples (own photos only) alongside the numbers.
- Tested only against SD1.5 img2img. Transfer to SDXL, InstructPix2Pix, inpainting and
  face-swap models is untested.
- Images are resized to max 512 px.
- Purification (e.g. DiffPure), heavy compression, and screenshots can remove the protection.
  This is an arms race, not a guarantee.

## Responsible use
Intended to help people protect their own photos. Do not publish AI edits of other people's
photos as demonstrations; use your own images for before/after examples.
