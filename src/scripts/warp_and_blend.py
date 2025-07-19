import cv2
import numpy as np
from PIL import Image
import json

def patch_and_blend_with_homography_skin_graft(original_path, enhanced_path, homography_path, output_path):
    original = np.array(Image.open(original_path).convert("RGB"))
    enhanced = np.array(Image.open(enhanced_path).convert("RGB"))

    with open(homography_path, 'r') as f:
        data = json.load(f)
        H_inv = np.array(data['H_inv'], dtype=np.float32)
        original_size = tuple(data['original_size'])

    H, W = original.shape[:2]
    warped_enhanced = cv2.warpPerspective(
        enhanced,
        H_inv,
        (W, H),  # width, height ordering!
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # Generate mask based on face region
    def generate_skin_mask(image):
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                SKIN_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 66,
                                107, 55, 65, 52, 53]
                points = [(int(l.x * w), int(l.y * h)) for i, l in enumerate(lm) if i in SKIN_INDICES]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        return (mask * 255).astype(np.uint8)

    mask = generate_skin_mask(original)

    # Extract detail map from warped enhanced image
    lab = cv2.cvtColor(warped_enhanced, cv2.COLOR_RGB2LAB)
    L, _, _ = cv2.split(lab)
    smooth_L = cv2.GaussianBlur(L, (0, 0), 2.5)
    detail_map = L.astype(np.float32) - smooth_L.astype(np.float32)

    # Inject detail back into original using mask
    lab_orig = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    L_orig, A_orig, B_orig = cv2.split(lab_orig)

    mask_bin = (mask > 150).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_bin, distanceType=cv2.DIST_L2, maskSize=5)
    max_dist = 20.0
    soft_mask = np.clip(dist_transform / max_dist, 0, 1).astype(np.float32)

    outer_mask_bin = (mask > 50).astype(np.uint8)
    dist_outer = cv2.distanceTransform(outer_mask_bin, distanceType=cv2.DIST_L2, maskSize=5)
    soft_outer = np.clip(dist_outer / (max_dist * 2), 0, 1).astype(np.float32)
    combined_mask = np.maximum(soft_mask, soft_outer * 0.5)

    L_detail = L_orig.astype(np.float32) + detail_map * combined_mask * 2  # strength=2
    L_detail = np.clip(L_detail, 0, 255).astype(np.uint8)

    result = cv2.merge([L_detail, A_orig, B_orig])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    Image.fromarray(result).save(output_path)
    print(f"✅ Saved blended skin-grafted result to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    parser.add_argument('--enhanced', required=True)
    parser.add_argument('--homography', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    patch_and_blend_with_homography_skin_graft(
        args.original,
        args.enhanced,
        args.homography,
        args.output
    )
#note: consider adding arcface to quality check the outputs