import json
import os
import cv2
import numpy as np

from lu_vp_detect import VPDetection

try:
    from .kd_tree import KdTree
except ImportError:
    from kd_tree import KdTree

def _rotate_with_padding(image, rotation_degrees, interpolation=cv2.INTER_LINEAR):
    """
    Rotate without clipping content, then fit back to original size (zoomed-out look).
    """
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])

    # Expanded canvas that fully contains the rotated image.
    bound_w = int(np.ceil((height * sin_a) + (width * cos_a)))
    bound_h = int(np.ceil((height * cos_a) + (width * sin_a)))

    # Shift rotation center to the center of the expanded canvas.
    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]

    rotated_full = cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Fit rotated full image into original dimensions to "zoom out" and avoid clipping.
    scale = min(width / float(bound_w), height / float(bound_h))
    fit_w = max(1, int(round(bound_w * scale)))
    fit_h = max(1, int(round(bound_h * scale)))
    fitted = cv2.resize(rotated_full, (fit_w, fit_h), interpolation=interpolation)

    canvas = np.zeros_like(image)
    x0 = (width - fit_w) // 2
    y0 = (height - fit_h) // 2
    canvas[y0:y0 + fit_h, x0:x0 + fit_w] = fitted
    return canvas


def _forward_affine_original_to_upright(image_shape, rotation_degrees):
    """
    Build the affine that maps original image coordinates to the final
    upright canvas coordinates produced by _rotate_with_padding.
    """
    height, width = image_shape[:2]
    center = (width / 2.0, height / 2.0)

    rot = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    cos_a = abs(rot[0, 0])
    sin_a = abs(rot[0, 1])

    bound_w = int(np.ceil((height * sin_a) + (width * cos_a)))
    bound_h = int(np.ceil((height * cos_a) + (width * sin_a)))

    rot[0, 2] += (bound_w / 2.0) - center[0]
    rot[1, 2] += (bound_h / 2.0) - center[1]

    scale = min(width / float(bound_w), height / float(bound_h))
    fit_w = max(1, int(round(bound_w * scale)))
    fit_h = max(1, int(round(bound_h * scale)))

    sx = fit_w / float(bound_w)
    sy = fit_h / float(bound_h)
    x0 = (width - fit_w) // 2
    y0 = (height - fit_h) // 2

    rot3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    fit3 = np.array(
        [
            [sx, 0.0, float(x0)],
            [0.0, sy, float(y0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    forward3 = fit3 @ rot3
    return forward3[:2, :]


def _choose_vertical_vp(vps_2d, principal_point):
    """
    Pick the vanishing point whose direction from principal point is closest to vertical.
    Returns None if no finite candidate is available.
    """
    cx, cy = principal_point

    candidates = []
    for idx, vp in enumerate(vps_2d):
        if vp is None or len(vp) < 2:
            continue

        x = float(vp[0])
        y = float(vp[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        dx = x - cx
        dy = y - cy
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            continue

        # Score by |cos(theta with vertical)|, higher is better.
        vertical_alignment = abs((-dy) / norm)
        candidates.append((vertical_alignment, idx, (x, y)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _rotation_to_make_vector_up(vector_xy):
    """
    Compute CV2 rotation angle (degrees) so the vector points to image up direction.
    """
    dx, dy = vector_xy

    # Convert image coordinates (y down) to Cartesian (y up) for angle math.
    cartesian_angle = np.degrees(np.arctan2(-dy, dx))
    return 90.0 - cartesian_angle


def _normalize_to_nearest_equivalent_rotation(angle_degrees):
    """
    Vanishing-point orientation is ambiguous up to 180 degrees.
    Keep the smallest equivalent rotation to avoid upside-down outputs.
    """
    return ((angle_degrees + 90.0) % 180.0) - 90.0


def rotate_image_upright_with_vp(
    image,
    length_thresh=60,
):
    height, width = image.shape[:2]

    principal_point = (width / 2.0, height / 2.0)

    focal_length = float(max(width, height))

    vpd = VPDetection(length_thresh, principal_point, focal_length)

    # Run VP detection.
    vps = vpd.find_vps(image)
    vps_2d = np.array(vpd.vps_2D, dtype=np.float64)

    chosen = _choose_vertical_vp(vps_2d, principal_point)
    if chosen is None:
        raise RuntimeError("No valid finite vanishing point found to estimate upright rotation.")

    vp_index, chosen_vp = chosen
    dx = chosen_vp[0] - principal_point[0]
    dy = chosen_vp[1] - principal_point[1]
    rotation_degrees = _rotation_to_make_vector_up((dx, dy))
    return _normalize_to_nearest_equivalent_rotation(rotation_degrees)

def _rotate_back_and_refill(image, rotation_degrees, interpolation=cv2.INTER_LINEAR, fill_image=None):
    """
    Inverse-warp upright image back to original orientation at original resolution
    without crop/resize zoom heuristics.
    """
    height, width = image.shape[:2]

    forward = _forward_affine_original_to_upright(image.shape, -rotation_degrees)
    forward3 = np.vstack([forward, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(forward3)[:2, :]

    restored = cv2.warpAffine(
        image,
        inverse,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if fill_image is None:
        return restored

    support = np.full((height, width), 255, dtype=np.uint8)
    valid_mask = cv2.warpAffine(
        support,
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask_3c = valid_mask[..., np.newaxis] > 0

    output = fill_image.copy()
    output[valid_mask_3c] = restored[valid_mask_3c]
    return output



class UprightKdTree(KdTree):
    """
    A wrapper around KdTree that applies VP-based upright rotation before building the tree.
    """

    def __init__(self, color_image, normal_image, depth_image, min_size=8, dot_threshold=0.98, **kwargs):
        self.rotation_degrees = rotate_image_upright_with_vp(color_image)

        rotated_image = _rotate_with_padding(color_image, self.rotation_degrees)

        rotated_normal = _rotate_with_padding(normal_image, self.rotation_degrees)

        rotated_depth = _rotate_with_padding(depth_image, self.rotation_degrees)

        super().__init__(
            color_image=rotated_image,
            normal_image=rotated_normal,
            depth_image=rotated_depth,
            min_size=min_size,
            dot_threshold=dot_threshold,
            **kwargs,
        )

    def rotate_back_and_refill(self, image):
        return _rotate_back_and_refill(image, -self.rotation_degrees)

    
