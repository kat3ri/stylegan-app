import torch
import numpy as np
from PIL import Image
import cv2
import sys
import argparse
import mediapipe as mp
import os
import sys


# Ensure src/ is in sys.path:
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_root)

#from stylegan2.model import Generator
from utils import legacy
from configs.paths_config import model_paths

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load pretrained architecture ===
STYLEGAN_CKPT = model_paths["STYLEGAN_CKPT"]
with open(STYLEGAN_CKPT, 'rb') as f:
    ckpt = legacy.load_network_pkl(f)
    g = ckpt['G_ema'].to(device).eval()
    mean_latent = ckpt.get('latent_avg', None)
    if mean_latent is None:
        z_samples = torch.randn(10000, g.z_dim, device=device)
        w_samples = g.mapping(z_samples, None)
        mean_latent = w_samples[:, 0, :].mean(0, keepdim=True)

mp_face_mesh = mp.solutions.face_mesh



def generate_variable_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return mask
        lm = results.multi_face_landmarks[0].landmark
        FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        skin_points = [(int(l.x * w), int(l.y * h)) for i, l in enumerate(lm) if i in FACE_OUTLINE]
        hull = cv2.convexHull(np.array(skin_points))
        cv2.fillConvexPoly(mask, hull, 1.0)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return (mask * 255).astype(np.uint8)

def downsample_mask(mask, target_size):
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA) / 255.0

def blend_latents_scaled(latent_clean, latent_detail, mask_img, global_strength):
    blended = []
    resolutions = {
        0: 4, 1: 4, 2: 8, 3: 8, 4: 16, 5: 16, 6: 32, 7: 32,
        8: 64, 9: 64, 10: 128, 11: 128, 12: 256, 13: 256,
        14: 512, 15: 512, 16: 512, 17: 512, 18: 256, 19: 256,
        20: 256, 21: 128, 22: 128, 23: 128, 24: 64, 25: 64
    }
    relative_factors = {l: 1.0 for l in resolutions.keys()}  # Optionally tweak by layer

    for i, (lc, ld) in enumerate(zip(latent_clean, latent_detail)):
        if i in resolutions:
            res = resolutions[i]
            mask_ds = downsample_mask(mask_img, (res, res))
            mask_mean = np.mean(mask_ds)
            factor = np.clip(mask_mean * relative_factors[i] * global_strength, 0.0, 1.0)
            blended_layer = lc * (1 - factor) + ld * factor
        else:
            blended_layer = lc
        blended.append(blended_layer.unsqueeze(0))

    return blended


def run_pipeline(image_path, latent_clean_path, latent_edited_path, output_path, global_strength):
    np_img = np.array(Image.open(image_path).convert("RGB"))
    mask = generate_variable_mask(np_img)

    latent_clean = torch.load(latent_clean_path, map_location=device)
    latent_edited = torch.load(latent_edited_path, map_location=device)

    assert isinstance(latent_clean, list) and isinstance(latent_edited, list), "Expected list of S-space tensors"

    # Blend S-space latents
    latent_blended = blend_latents_scaled(latent_clean, latent_edited, mask, global_strength)

    # Prepare for synthesis:
    latent_blended_for_synthesis = [b.squeeze(0).squeeze(0) if b.ndim == 4 else b.squeeze(0) for b in latent_blended]

    dummy_ws = torch.zeros([1, g.num_ws, g.w_dim], device=device)
    
    print(f"[DEBUG] Blended latent layers: {len(latent_blended_for_synthesis)}")
    for i, l in enumerate(latent_blended_for_synthesis):
        print(f"[DEBUG] Layer {i} shape: {l.shape}")

    # Synthesize image:
    with torch.no_grad():
        blended_img = g.synthesis(
            dummy_ws,  # Mandatory ws arg
            encoded_styles=latent_blended_for_synthesis,
            noise_mode='const'
        )

        

    img_out = blended_img[0].cpu().permute(1, 2, 0)
    img_out = (img_out * 127.5 + 128).clamp(0, 255).byte().numpy()
    Image.fromarray(img_out, mode='RGB').save(output_path)
    print(f"✅ Saved blended output image to {output_path}")

    latent_save_path = output_path.replace(".png", "_latent.pt")
    torch.save(latent_blended, latent_save_path)
    print(f"✅ Saved blended latent to {latent_save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct blending of clean + edited latents with mask guidance.")
    parser.add_argument("--input", required=True, help="Path to input image for mask generation.")
    parser.add_argument("--latent_clean", required=True, help="Path to clean latent.pt file.")
    parser.add_argument("--latent_edited", required=True, help="Path to edited latent.pt file.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--strength", type=float, default=1.0, help="Global strength multiplier.")
    args = parser.parse_args()

    run_pipeline(args.input, args.latent_clean, args.latent_edited, args.output, args.strength)