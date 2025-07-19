import subprocess
import argparse
import os
import json
from datetime import datetime
import shutil
from configs import paths_config
from configs.paths_config import model_paths

# ------------
# Defaults
# ------------
DEFAULT_NUM_THREADS = 4
ROOT_PATH = os.getcwd()  # Automatically use project root as base path
WORKSPACE_DIR = "./workspace"  # Common parent folder for all runs



def run_align_faces(input_dir, num_threads):
    cmd = [
        "python", "scripts/align_faces_parallel.py",
        "--num_threads", str(num_threads),
        "--root_path", input_dir
    ]
    subprocess.run(cmd, check=True)

def run_inference(checkpoint_path, aligned_dir, exp_dir, n_iters, test_batch_size):
    cmd = [
        "python", "scripts/inference.py",
        "--checkpoint_path", checkpoint_path,
        "--data_path", aligned_dir,
        "--exp_dir", exp_dir,
        "--n_iters", str(n_iters),
        "--test_batch_size", str(test_batch_size),
        "--load_w_encoder",
        "--save_weight_deltas"
    ]
    subprocess.run(cmd, check=True)

def run_generate_edit_latents(image_path, latent_path, output_image, strength):
    cmd = [
        "python", "scripts/generate_edit_latents.py",
        "--image", image_path,
        "--latent", latent_path,
        "--output", output_image,
        "--strength", str(strength)
    ]
    subprocess.run(cmd, check=True)

def run_warp_and_blend(original_path, enhanced_path, homography_path, final_output):
    cmd = [
        "python", "scripts/warp_and_blend.py",
        "--original", original_path,
        "--enhanced", enhanced_path,
        "--homography", homography_path,
        "--output", final_output
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end workflow runner with automatic directory layout")
    parser.add_argument('--image_path', type=str, required=True, help='Path to original input image')
    parser.add_argument('--num_threads', type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument('--n_iters', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--run_id', type=str, default=None, help='Optional run ID; timestamp if not provided')

    args = parser.parse_args()

    # Load config

    checkpoint_path = paths_config.model_paths["checkpoint_path"]


    # Generate unique run ID
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(WORKSPACE_DIR, f"run_{run_id}")

    # Define structured subfolders for this run
    input_dir = os.path.join(run_dir, "input")
    output_dir = os.path.join(run_dir, "output")
    aligned_dir = os.path.join(input_dir, "aligned")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Copy input image into input_dir
    input_image_name = os.path.basename(args.image_path)
    run_image_path = os.path.join(input_dir, input_image_name)
    shutil.copy2(args.image_path, run_image_path)

    latent_path = os.path.join(output_dir, "latents.npy")
    homography_path = run_image_path + "_homography.json"
    blended_output = os.path.join(output_dir, "blended_output.png")
    final_output = os.path.join(output_dir, "final_skin_graft.png")

    print(f"➡️ Starting workflow run {run_id} in {run_dir}")

    print("➡️ Running align_faces_parallel.py...")
    run_align_faces(input_dir, args.num_threads)

    print("➡️ Running inference.py...")
    run_inference(checkpoint_path, aligned_dir, output_dir, args.n_iters, args.test_batch_size)

    print("➡️ Running generate_edit_latents.py...")
    run_generate_edit_latents(run_image_path, latent_path, blended_output, args.strength)

    print("➡️ Running warp_and_blend.py...")
    run_warp_and_blend(
        run_image_path,
        blended_output,
        homography_path,
        final_output
    )

    print(f"✅ Workflow complete. Final output saved at: {final_output}")
