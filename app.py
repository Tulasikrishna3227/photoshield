import gradio as gr

from protect import protect

NOTE = ("Protection is partial, not a guarantee. It targets Stable Diffusion based editors and can be "
        "weakened by heavy compression, resizing, screenshots or purification. Download the PNG; "
        "re-saving as JPEG reduces the effect. Images are resized to at most 512 px.")
STRENGTH = {"Standard (6/255)": 6, "Strong (8/255)": 8, "Maximum (12/255)": 12}


def run(img, strength, steps, robust, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Upload an image first.")
    out = protect(img, eps=STRENGTH[strength] / 255, steps=int(steps),
                  robust=bool(robust), progress=progress)
    out.save("protected.png", format="PNG")
    return out, "protected.png"


with gr.Blocks(title="PhotoShield") as demo:
    gr.Markdown("# PhotoShield\nInvisible protection that makes Stable Diffusion edits of your photo "
                "come out distorted.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Original photo")
        out = gr.Image(type="pil", label="Protected photo")
    strength = gr.Radio(list(STRENGTH), value="Strong (8/255)", label="Strength")
    steps = gr.Slider(20, 200, value=100, step=10, label="Optimization steps (more = stronger, slower)")
    robust = gr.Checkbox(value=True, label="Robust mode (recommended: survives JPEG/resize/blur better)")
    btn = gr.Button("Protect", variant="primary")
    file = gr.File(label="Download PNG")
    gr.Markdown(NOTE)
    btn.click(run, [inp, strength, steps, robust], [out, file])

if __name__ == "__main__":
    demo.launch()
