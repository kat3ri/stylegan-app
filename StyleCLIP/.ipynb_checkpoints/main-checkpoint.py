import argparse
import os
import subprocess
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from gdown import download as drive_download
import clip
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import sys
import cv2
import json

from encoder4editing.utils.common import tensor2im
from encoder4editing.models.psp import pSp
from global_torch.manipulate import Manipulator
from global_torch.StyleCLIP import GetDt, GetBoundary

sys.path.insert(0, '/workspace/stylegan-app/src')
from configs.paths_config import model_paths

# -------------------------
# CLI ARGUMENT PARSER
# -------------------------
parser = argparse.ArgumentParser(description="StyleCLIP local runner")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
#parser.add_argument("--experiment_type", type=str, default="ffhq_encode", help="Experiment type")
parser.add_argument("--strength", type=float, default=2.0, help="Global strength multiplier for generate_edit_latents")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output image")
args = parser.parse_args()

image_path = args.image_path
strength = args.strength

# -------------------------
# Workspace / run layout config
# -------------------------
WORKSPACE_DIR = "./workspace"
run_id = datetime.now().strftime("%Y%m%d%H%M%S")
run_dir = os.path.join(WORKSPACE_DIR, f"run_{run_id}")

input_dir = os.path.join(run_dir, "input")
output_dir = os.path.join(run_dir, "output")
aligned_dir = os.path.join(input_dir, "aligned")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Copy input image into input_dir for consistency
# -------------------------
input_image_name = os.path.basename(image_path)
run_image_path = os.path.join(input_dir, input_image_name)
shutil.copy2(image_path, run_image_path)
print(f"[DEBUG] Copied input image to {run_image_path}")

# -------------------------
# All subsequent paths should use run_dir structure:
LATENTS_PATH = os.path.join(output_dir, "latents.pt")
EDITED_LATENTS_PATH = os.path.join(output_dir, "edited_latents.pt")
INVERSION_RESULT_PATH = os.path.join(output_dir, "inversion_result.png")
FINAL_EDIT_RESULT_PATH = os.path.join(output_dir, "final_edit_result.png")


# -------------------------
# CONFIG
# -------------------------
dataset_name = 'ffhq'
checkpoint_local_path = './encoder4editing/e4e_ffhq_encode.pt'
experiment_type = 'ffhq_encode'
model_dir = './global_torch/model/'
model_path = os.path.join(model_dir, f'{dataset_name}.pkl')

ALIGN_SCRIPT_PATH = '/workspace/stylegan-app/src/scripts/align_faces_parallel.py'
ALIGNED_SUBDIR = 'aligned'
STYLEGAN_APP_ROOT = '/workspace/stylegan-app'
INPUTS_DIR = os.path.join(STYLEGAN_APP_ROOT, 'inputs')
resize_dims = (256, 256)

aligned_dir = os.path.join(os.path.dirname(image_path), ALIGNED_SUBDIR)
aligned_image_path = os.path.join(aligned_dir, os.path.basename(image_path))
# If image_path came from CLI as relative:
if not os.path.isabs(image_path):
    # Resolve it relative to INPUTS_DIR explicitly
    image_path = os.path.abspath(os.path.join(INPUTS_DIR, os.path.basename(image_path)))

print(f"[DEBUG] Resolved image_path: {image_path}")

latent_path = './edited_latents.pt'
output_img_path = './final_edit_result.png'

# -------------------------
# Ensure working directories exist
# -------------------------
os.makedirs(model_dir, exist_ok=True)


# -------------------------
# Download pretrained model if missing
# -------------------------
if not os.path.isfile(model_path):
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
    name = f'stylegan2-{dataset_name}-config-f.pkl'
    print(f"[DEBUG] Downloading {url}{name} ...")
    os.system(f'wget {url}{name} -P {model_dir}')
    os.rename(os.path.join(model_dir, name), model_path)
    print(f"[DEBUG] Model downloaded to {model_path}")

if not os.path.isfile(checkpoint_local_path):
    print("[DEBUG] Downloading e4e checkpoint...")
    drive_download(
        "https://drive.google.com/uc?id=1O8OLrVNOItOJoNGMyQ8G8YRTeTYEfs0P",
        checkpoint_local_path,
        quiet=False
    )


# -------------------------
# Initialize CLIP model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# -------------------------
# Manipulator workflow
# -------------------------
M = Manipulator()
M.device = device

G = M.LoadModel(model_path, device)
M.G = G
M.SetGParameters()

num_img = 100_000
M.GenerateS(num_img=num_img)
M.GetCodeMS()

np.set_printoptions(suppress=True)

file_path = f'./global_torch/npy/{dataset_name}/'
fs3 = np.load(file_path + 'fs3.npy')

# -------------------------
# e4e setup
# -------------------------
EXPERIMENT_ARGS = {
    "model_path": checkpoint_local_path,
    "transform": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

ckpt = torch.load(EXPERIMENT_ARGS['model_path'], map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = checkpoint_local_path
opts = Namespace(**opts)

net = pSp(opts)
net.eval()
net.cuda()

print('[DEBUG] e4e model successfully loaded!')

# -------------------------
# Alignment function
# -------------------------
def run_alignment(image_path, num_threads=4):
    image_path = os.path.abspath(image_path)
    root_path = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)

    cmd = [
        'python', ALIGN_SCRIPT_PATH,
        '--root_path', root_path,
        '--num_threads', str(num_threads)
    ]
    print(f"[DEBUG] Running alignment script: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("Alignment script failed")

    aligned_dir = os.path.join(root_path, ALIGNED_SUBDIR)
    aligned_image_path = os.path.join(aligned_dir, image_filename)
    if not os.path.isfile(aligned_image_path):
        raise FileNotFoundError(f"Aligned image not found at {aligned_image_path}")

    aligned_image = Image.open(aligned_image_path).convert("RGB")
    print(f"[DEBUG] Aligned image size: {aligned_image.size}")
    return aligned_image

# -------------------------
# Inversion utilities
# -------------------------


def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([
        np.array(source_image.resize(resize_dims)),
        np.array(result_image.resize(resize_dims))
    ], axis=1)
    return Image.fromarray(res)

def run_on_batch(inputs, net):
    images, latents = net(inputs.to(device).float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents

#Blending logic

def run_warp_and_blend(original_path, enhanced_path, homography_path, final_output_path, strength):
    original = os.path.abspath(original_path)
    enhanced = os.path.abspath(enhanced_path)
    homography = os.path.abspath(homography_path)
    output = os.path.abspath(final_output_path)

    cmd = [
        'python', WARP_AND_BLEND_SCRIPT_PATH,  # <-- ensure this path is correct
        '--original', original_path,
        '--enhanced', enhanced_path,
        '--homography', homography_path,
        '--output', final_output_path,
        '--strength', str(args.strength)
    ]

    print(f"[DEBUG] Running warp_and_blend.py with args: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[DEBUG] Final blended output saved to {final_output_path}")




# -------------------------
# Align or load image
# -------------------------
root_path_for_alignment = input_dir
if experiment_type == "ffhq_encode":
    input_image = run_alignment(run_image_path)
else:
    input_image = Image.open(run_image_path).convert("RGB")

input_image = input_image.resize(resize_dims)
print("[DEBUG] Input image prepared.")


# -------------------------
# Run inversion
# -------------------------
img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)

with torch.no_grad():
    images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
    result_image = images[0]

print("[DEBUG] Inversion complete")

# Optional debug view:
final_display_image = display_alongside_source_image(
    tensor2im(result_image),
    input_image
)
#final_display_image.save("inversion_result.png")
#print("[DEBUG] Inversion result saved")

# -------------------------
# Manipulation setup
# -------------------------
latents = latents.to(device)
dlatents_loaded = M.G.synthesis.W2S(latents)
dlatents_loaded = M.S2List(dlatents_loaded)
dlatent_tmp = [tmp[[0]] for tmp in dlatents_loaded]  # ensure shape [1, ...]

M.num_images = 1
M.alpha = [0]
M.manipulate_layers = [0]

codes, out = M.EditOneC(0, dlatent_tmp)
original = Image.fromarray(out[0, 0]).resize((512, 512))
M.manipulate_layers = None
print("[DEBUG] Original reconstruction generated")

# -------------------------
# Apply CLIP-based edit
# -------------------------
neutral = 'face with clear skin'
target = 'face with pores, skin texture, and blemishes'
classnames = [target, neutral]

dt = GetDt(classnames, model)
print("[DEBUG] Directional text embedding computed")

beta = 0.1
alpha = 4.0

M.alpha = [alpha]
boundary_tmp2, _ = GetBoundary(fs3, dt, M, threshold=beta)
codes = M.MSCode(dlatent_tmp, boundary_tmp2)
out = M.GenerateImg(codes)
generated = Image.fromarray(out[0, 0])#.resize((512, 512))
print("[DEBUG] Manipulated image generated")

manipulated_image_path = os.path.join(output_dir, "manipulation_result.png")
generated.save(manipulated_image_path)
print(f"[DEBUG] Manipulation result saved at {manipulated_image_path}")

# -------------------------
# Warp and blend result directly — no latent blending!
# -------------------------
WARP_AND_BLEND_SCRIPT_PATH = '/workspace/stylegan-app/src/scripts/warp_and_blend.py'
homography_path = run_image_path + "_homography.json"
final_output_path = os.path.join(output_dir, "final_skin_graft.png")

run_warp_and_blend(
    original_path=run_image_path,
    enhanced_path=manipulated_image_path,
    homography_path=homography_path,
    final_output_path=final_output_path,
    strength=args.strength
)
print(f"[DEBUG] Final warp-and-blend complete at {final_output_path}")


