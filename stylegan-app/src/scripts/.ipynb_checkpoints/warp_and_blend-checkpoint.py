import cv2
import numpy as np
from PIL import Image
import json

def patch_and_blend_with_homography_skin_graft(original_path, enhanced_path, homography_path, output_path, strength):
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
    def generate_skin_mask(image, exclusion_padding=10):
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
    
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
    
                # Expanded skin mask indices to include upper forehead
                SKIN_INDICES = [
                    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 
                    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
                    234, 127, 162, 21, 54, 103, 67, 109, 66, 107, 55, 65, 52, 53, 67, 69, 
                    108, 151, 337, 9, 8, 107  # 9, 8 add forehead top
                ]
    
                points = [(int(l.x * w), int(l.y * h)) for i, l in enumerate(lm) if i in SKIN_INDICES]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 1.0)

              # Exclude regions with optional padding (eyes, mouth, nose)
                def erase_region(indices, padding=exclusion_padding):
                    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in indices], np.int32)
                    if len(pts) == 0:
                        return
                    rect = cv2.boundingRect(pts)
                    center = (rect[0] + rect[2]//2, rect[1] + rect[3]//2)
                    pts = pts - center
                    scale = 1 + padding / 100.0
                    pts = pts * scale
                    pts = pts + center
                    pts = pts.astype(np.int32)
                    cv2.fillConvexPoly(mask, pts, 0.0)
    
                left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                         78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
                nose = [1, 2, 98, 327, 195, 5, 4, 275, 440]
                upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

    
                erase_region(left_eye)
                erase_region(right_eye)
                erase_region(mouth, padding=10)
                erase_region(nose, padding=5)
                erase_region(upper_lip, padding=5)  # smaller padding to preserve lower lip but protect upper

                # === NEW: Forehead center guided soft blend ===
                anchor = lm[10]  # Between eyebrows
                fx = int(anchor.x * w)
                fy = int(anchor.y * h)
    
                forehead_radius = int(0.2 * h)
                max_y = max(fy - int(0.25 * h), 0)  # Limit upward extent
    
                for y in range(max_y, fy):
                    for x in range(w):
                        dx = abs(x - fx)
                        dy = fy - y
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist < forehead_radius:
                            blend_val = 1.0 - (dist / forehead_radius)
                            mask[y, x] = max(mask[y, x], blend_val * 0.8)  # Soft blend at 80% opacity max


    
        mask = cv2.GaussianBlur(mask, (41, 41), 0)  # Moderate blur
        return (mask * 255).astype(np.uint8)



    mask = generate_skin_mask(original)
#    cv2.imwrite("DEBUG_MASK.png", mask)
    print(f"[DEBUG] Mask mean: {mask.mean()}, max: {mask.max()}")

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

    L_detail = L_orig.astype(np.float32) + detail_map * combined_mask * strength
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
    parser.add_argument('--strength', type=float, default=2.0, help='Detail enhancement strength')
    args = parser.parse_args()

    patch_and_blend_with_homography_skin_graft(
        args.original,
        args.enhanced,
        args.homography,
        args.output,
        args.strength
    )
#note: consider adding arcface to quality check the outputs