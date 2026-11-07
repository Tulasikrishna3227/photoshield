# PhotoShield: adversarial perturbations against Stable Diffusion photo editing

An implementation and an honest evaluation of an encoder-level adversarial perturbation
(PhotoGuard-style) meant to make Stable Diffusion edits of a photo fail. It adds an invisible,
texture-masked perturbation to a photo; an optional robust mode optimises through random
JPEG / resize / blur.

## Summary of findings
- **Works against low-strength edits on unprocessed images.** SD1.5 img2img at strength 0.3:
  the edit of the protected photo drifts away from the source on 15 of 16 test images, versus
  0 of 16 for random noise of the same size.
- **Does not stop scene-changing edits.** At strength 0.6 the edits of protected photos follow
  the prompt as well as edits of unprotected ones (CLIP test), no better than the noise control.
- **Fragile to post-processing.** After JPEG q75 or a 50% resize, plain protection falls to
  near noise level (0 to 1 of 16 images). Robust training recovers part of it (robust eps 12:
  6/16 after JPEG, 5/16 after resize), at the cost of visibility (about 35 dB PSNR in an
  8-image test).
- **My first metric overstated the results.** LPIPS between the edit of the original and the
  edit of the protected image only shows that the edits differ, and random noise makes them
  differ too. I replaced it with a drift-from-source metric and a noise control.

## Run
    pip install -r requirements.txt
    python app.py                                   # Gradio demo
    python protect.py in.jpg out.png [--robust]     # CLI; always save PNG
    python strength_eval.py --images ./test_images --strengths 0.3 --attacks none,jpeg75,resize50
    python clip_eval.py     --images ./test_images --n 16
    python make_figure.py   --image my_photo.jpg --out docs/demo.png   # use a photo you own

## Method
1. Encode a high-frequency checkerboard with the SD VAE to get a "garbage" target latent.
2. PGD (100 steps) finds delta so VAE(x + delta) matches it, with |delta| <= eps * mask.
3. The mask is high on textured areas and low on flat ones (walls, skin), which removed the
   visible blotches a uniform eps produced.
4. Robust mode: each PGD step passes x + delta through a random differentiable transform
   (JPEG approximation via 8x8 DCT with straight-through rounding, resize, Gaussian blur).

## Evaluation protocol
- **Edit model:** SD1.5 img2img, 30 steps, guidance 7.5, fixed seed.
- **Primary metric, shift:** LPIPS(edit(variant), original) - LPIPS(edit(original), original).
  At low strength an edit of an unprotected photo stays close to the photo; a working protection
  makes the edit drift or break.
- **Control:** random +-eps sign noise with the same texture mask and eps. Control shift is
  0.006 to 0.018, so a shift above 0.10 is clearly beyond noise (threshold chosen before
  seeing the data).
- **CLIP prompt adherence** for strength 0.6: drop in CLIP(prompt, edit) versus the edit of the
  original.
- **Test set:** 16 images (15 web-sourced portraits, not redistributed, and 1 personal photo).
  Image names in the CSVs are anonymised.
- **Raw data:** per-image CSVs are included for the ablations (`ablation_v2.csv`, `ablation_v3.csv`,
  `results_v2_eps6.csv`) and the strength 0.3/0.5 run (`strength_eval.csv`). The post-processing
  run and the CLIP run were reported from console output; their per-image CSVs were not saved
  because the compute session reset before download.

## Results
### Low-strength img2img, unprocessed images (mean shift; images with shift > 0.10 of 16)
| Variant | strength 0.3 | strength 0.5 | PSNR |
|---|---|---|---|
| random-noise control (eps 8) | 0.018 (0) | 0.010 (0) | 37.2 |
| plain eps 8 | 0.185 (14) | 0.150 (12) | 38.0 |
| robust eps 8 | 0.125 (11) | 0.106 (7) | 38.3 |
| plain eps 16, 200 steps | 0.296 (16) | 0.210 (14) | 32.5 |

### After post-processing, strength 0.3
| Variant | none | JPEG q75 | resize 50% |
|---|---|---|---|
| random-noise control (eps 8) | 0.018 (0) | 0.006 (0) | 0.007 (0) |
| plain eps 8 | 0.182 (15) | 0.027 (0) | 0.029 (1) |
| robust eps 8 | 0.128 (9) | 0.059 (2) | 0.040 (1) |
| robust eps 12 | 0.188 (15) | 0.094 (6) | 0.069 (5) |

Run-to-run GPU nondeterminism explains the small difference between the two plain eps 8 rows.

### Scene-changing edits, strength 0.6 (mean CLIP prompt-adherence drop; higher = more broken)
| Variant | none | JPEG q75 |
|---|---|---|
| random-noise control (eps 8) | -0.017 | -0.005 |
| plain eps 8 | -0.018 | -0.014 |
| robust eps 8 | -0.025 | -0.019 |
| plain eps 16, 200 steps | -0.029 | -0.010 |

Only 0 to 2 of 16 images show a drop above 0.05 (control: 0). Protected edits adhere to the
prompt as well as unprotected ones.

### Earlier ablations (kept for transparency)
`ablation_v2.csv` and `ablation_v3.csv` used LPIPS between edits at strength 0.6 and are
superseded: that metric could not separate real protection from noise. The v3 finding that
robust training raises that score after JPEG/resize is consistent with the post-processing
table above, but the absolute claims in those tables should not be read as "edits destroyed".

## Limitations
- Small evaluation: 16 images, one prompt per experiment, one seed, no confidence intervals.
- Shift measures how far the edit drifts from the source, not how broken it looks. Visual
  inspection of a few own photos is needed alongside the numbers.
- Tested only against SD1.5 img2img. Inpainting, InstructPix2Pix, SDXL and face-swap models
  are untested.
- Images are resized to at most 512 px.
- Not a guarantee. Re-saving, resizing, screenshots and purification (e.g. DiffPure) weaken or
  remove the effect, and newer models may ignore it.

## Next steps
1. A denoiser-level attack (Mist / PhotoGuard diffusion attack) instead of encoder-only.
2. Inpainting and InstructPix2Pix evaluation; SDXL transfer.
3. More images, prompts and seeds with confidence intervals.
4. A blind visual rating of "broken vs normal" edits.

## Responsible use
Intended to help people protect their own photos. Do not publish AI edits of other people's
photos as demonstrations; use your own images for before/after examples.
