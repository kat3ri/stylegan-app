import gradio as gr
import subprocess
import tempfile
import os
from datetime import datetime
from PIL import Image
import re


WORKSPACE_DIR = "./workspace"

def enhance_skin(image, strength):
    input_image_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(input_image_temp.name)

    completed_process = subprocess.run([
        "python", "-m", "main",
        "--image_path", input_image_temp.name,
        "--strength", str(strength),
        "--output_dir", WORKSPACE_DIR
    ], check=True, capture_output=True, text=True)

    os.remove(input_image_temp.name)

    output_match = re.search(r"Saved blended skin-grafted result to (.+final_skin_graft.png)", completed_process.stdout)
    if not output_match:
        raise RuntimeError("Failed to locate the output image path from main.py output.")

    final_output_path = output_match.group(1)
    result_image = Image.open(final_output_path)

    return result_image

interface = gr.Interface(
    fn=enhance_skin,
    inputs=[
        gr.Image(type="pil", label="Upload Face Image (.jpg, .jpeg, .png)"),
        gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=1.0, label="Strength")
    ],
    outputs=gr.Image(type="pil", label="Enhanced Image"),
    title="Skin Detailing App",
    description="Upload an image and adjust the strength for skin detailing."
)


if __name__ == "__main__":
    interface.launch(share=True)
