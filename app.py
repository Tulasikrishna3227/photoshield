import gradio as gr
from PIL import Image

from protect import protect

NOTE = ("Protection is not a guarantee. It can be weakened by heavy JPEG compression, "
        "resizing, screenshots, or purification methods. Always download the PNG.")


def run(img, strength, steps, robust, progress=gr.Progress()):  # robust = survives JPEG/resize better
    if img is None:
        raise gr.Error("Upload an image first.")
    eps = {"Low (4/255)": 4, "Medium (8/255)": 8, "High (12/255)": 12}[strength] / 255
    out = protect(img, eps=eps, steps=int(steps), robust=bool(robust), progress=progress)
    out.save("protected.png", format="PNG")
    return out, "protected.png"


with gr.Blocks(title="PhotoShield") as demo:
    gr.Markdown("# PhotoShield\nInvisible protection that makes AI edits of your photo come out broken.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Original photo")
        out = gr.Image(type="pil", label="Protected photo")
    strength = gr.Radio(["Low (4/255)", "Medium (8/255)", "High (12/255)"],
                        value="Medium (8/255)", label="Strength")
    steps = gr.Slider(20, 200, value=100, step=10, label="Optimization steps")
    robust = gr.Checkbox(label="Robust mode (optimised through JPEG/resize/blur)")
    btn = gr.Button("Protect", variant="primary")
    file = gr.File(label="Download PNG")
    gr.Markdown(NOTE)
    btn.click(run, [inp, strength, steps, robust], [out, file])

if __name__ == "__main__":
    demo.launch()
