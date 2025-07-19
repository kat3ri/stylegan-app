import torch
import numpy as np
from PIL import Image
import cv2
import sys
import argparse
import mediapipe as mp
import os


from stylegan2.model import Generator
from stylegan2_ada_pytorch import legacy
from configs.paths_config import model_paths

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load pretrained architecture ===
STYLEGAN_CKPT = model_paths["STYLEGAN_CKPT"]
pca = model_paths["pca"]

with open(STYLEGAN_CKPT, 'rb') as f:
    ckpt = legacy.load_network_pkl(f)
    g = ckpt['G_ema'].to(device).eval()
    if 'latent_avg' in ckpt:
        mean_latent = ckpt['latent_avg'].to(device)
    else:
        # Compute mean_latent if not present
        z_samples = torch.randn(10000, g.z_dim, device=device)
        w_samples = g.mapping(z_samples, None)
        mean_latent = w_samples[:, 0, :].mean(0, keepdim=True)


# === Compute mean_latent manually ===
z_samples = torch.randn(10000, g.z_dim, device=device)
w_samples = g.mapping(z_samples, None)
mean_latent = w_samples[:, 0, :].mean(0, keepdim=True)

g.eval()

pca = torch.load(pca)
mp_face_mesh = mp.solutions.face_mesh

def apply_ganspace_direction(latent, component_idx, strength=2.0):
    latent_out = latent.clone()
    U = pca['comp'].to(device)
    std = pca['std'].to(device) if isinstance(pca['std'], torch.Tensor) else torch.from_numpy(pca['std']).to(device)

    for layer in range(14, 18):
        direction = U[layer, :, component_idx]
        direction = direction * std[component_idx]
        latent_out[:, layer] = latent_out[:, layer] + strength * direction

    return latent_out



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
    resolutions = {14: 64, 15: 128, 16: 256, 17: 512}
    relative_factors = {14: 1.0, 15: 0.8, 16: 0.8, 17: 0.8}
    for i, (lc, ld) in enumerate(zip(latent_clean[0], latent_detail[0])):
        if 14 <= i <= 17:
            res = resolutions[i]
            mask_ds = downsample_mask(mask_img, (res, res))
            mask_mean = np.mean(mask_ds)
            factor = np.clip(mask_mean * relative_factors[i] * global_strength, 0.0, 1.0)
            blended_layer = lc * (1 - factor) + ld * factor
        else:
            blended_layer = lc
        blended.append(blended_layer.unsqueeze(0))
    return torch.cat(blended, dim=0).unsqueeze(0)

def run_pipeline(image_path, latent_path, output_path, global_strength):
    np_img = np.array(Image.open(image_path).convert("RGB"))
    mask = generate_variable_mask(np_img)

    latent_clean = None
    if latent_path.endswith('.npz'):
        npz = np.load(latent_path)
        latent_clean = torch.from_numpy(npz['w']).to(device)
    elif latent_path.endswith('.npy'):
        npy = np.load(latent_path, allow_pickle=True).item()  # Load dict
        latent_np = npy['face.jpg']  # E.g. 'face.jpg'
        #Note: need to fix this to add key dynamically
        # Ensure shape [1, 18, 512]
        latent_clean = torch.from_numpy(latent_np).unsqueeze(0).to(device)


    else:
        latent_clean = torch.load(latent_path, map_location=device)

    latent_detail = apply_ganspace_direction(latent_clean, component_idx=7, strength=1.5)
    for layer in range(14, 18):
        latent_detail[:, layer] += torch.randn_like(latent_detail[:, layer]) * 0.05

    latent_blended = blend_latents_scaled(latent_clean, latent_detail, mask, global_strength)

    if latent_blended.ndim == 2:
        w = latent_blended.unsqueeze(0)
    else:
        w = latent_blended

    trunc = 0.7
    w = mean_latent + trunc * (w - mean_latent)

    with torch.no_grad():
        blended_img = g.synthesis(w, noise_mode='const')

    img_out = blended_img[0].cpu().permute(1, 2, 0)
    img_out = (img_out * 127.5 + 128).clamp(0, 255).byte().numpy()
    Image.fromarray(img_out, mode='RGB').save(output_path)
    print(f"✅ Saved preview image to {output_path}")

    latent_save_path = output_path.replace(".png", "_latent.npz")
    np.savez(latent_save_path, w=latent_blended.detach().cpu().numpy())
    print(f"✅ Saved blended latent to {latent_save_path}")

def save_latent_preview(latent, path):
    if latent.ndim == 2:
        w = latent.unsqueeze(0)
    else:
        w = latent

    with torch.no_grad():
        img = g.synthesis(w, noise_mode='const')

    img_out = (img[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
    img_out = img_out.to(torch.uint8).cpu().numpy()
    Image.fromarray(img_out, 'RGB').save(path)
    print(f"✅ Saved exact latent preview to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent-space selective blending inspired by StyleRetoucher.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--latent", required=True, help="Path to latent.pt, .npz or .npy file.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--strength", type=float, default=1.0, help="Global strength multiplier.")
    args = parser.parse_args()

    run_pipeline(args.image, args.latent, args.output, args.strength)