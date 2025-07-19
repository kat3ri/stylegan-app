from argparse import ArgumentParser
import time
import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
import multiprocessing as mp
import math
import sys
import json
import cv2

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
SHAPE_PREDICTOR_PATH = model_paths["shape_predictor"]

def get_landmark(filepath, predictor):
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    t = list(shape.parts())
    lm = np.array([[tt.x, tt.y] for tt in t])
    return lm

def align_face(filepath, predictor):
    lm = get_landmark(filepath, predictor)
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    img = PIL.Image.open(filepath).convert("RGB")
    original_size = img.size
    output_size = 1024

    src_pts = quad.astype(np.float32)
    dst_pts = np.array([[0, 0], [0, output_size - 1], [output_size - 1, output_size - 1], [output_size - 1, 0]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    H_inv = np.linalg.inv(H)

    img_np = np.array(img)
    aligned = cv2.warpPerspective(img_np, H, (output_size, output_size), flags=cv2.INTER_LANCZOS4)

    # Save inverse homography for warp-back
    homography_path = filepath + '.homography.json'
    with open(homography_path, 'w') as f:
        json.dump({'H_inv': H_inv.tolist(), 'original_size': original_size}, f)

    return PIL.Image.fromarray(aligned)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_on_paths(file_paths):
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    for file_path, res_path in file_paths:
        try:
            res = align_face(file_path, predictor)
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            res.save(res_path)
        except Exception as e:
            print(f"Failed on image: {file_path} - {e}")
            continue

def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--root_path', type=str, default='')
    return parser.parse_args()

def run(args):
    root_path = args.root_path
    out_crops_path = os.path.join(root_path, 'aligned')
    os.makedirs(out_crops_path, exist_ok=True)

    file_paths = []
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        if os.path.isdir(file_path):
            continue

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            continue

        fname = os.path.join(out_crops_path, os.path.relpath(file_path, root_path))
        res_path = '{}.jpg'.format(os.path.splitext(fname)[0])

        file_paths.append((file_path, res_path))

    if len(file_paths) == 0:
        print(f"❌ No valid input images found in {root_path}")
        return

    file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
    pool = mp.Pool(args.num_threads)
    tic = time.time()
    pool.map(extract_on_paths, file_chunks)
    toc = time.time()
    print('Done in {}s'.format(toc - tic))

if __name__ == '__main__':
    args = parse_args()
    run(args)