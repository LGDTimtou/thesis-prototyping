import argparse
import cv2
import numpy as np
import os
import json
import sys
import importlib
import warnings
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trees.quadtree import Quadtree
from trees.kd_tree import KdTree
from trees.bsp_tree import BSPTree
from trees.upright_kd_tree import UprightKdTree
from trees.tree import Tree


VALID_TREE_NAMES = {
    "quadtree",
    "kdtree",
    "bsptree",
    "upright_kdtree",
}


def _compute_ssim_bgr(img_a_bgr, img_b_bgr):
    gray_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(gray_a, gray_b, data_range=255))


def _to_lpips_tensor_from_bgr(image_bgr, torch_module):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch_module.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def _compute_lpips_if_available(img_a_bgr, img_b_bgr, device="cpu"):
    """
    Try to compute LPIPS (alex). Returns (value_or_none, status_message).
    """
    try:
        if importlib.util.find_spec("torch") is None:
            return None, "torch not installed"
        if importlib.util.find_spec("lpips") is None:
            return None, "lpips not installed"

        import torch
        import lpips

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*parameter 'pretrained' is deprecated since 0.13.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
                category=UserWarning,
            )
            loss_fn = lpips.LPIPS(net="alex").to(device)
        t0 = _to_lpips_tensor_from_bgr(img_a_bgr, torch).to(device)
        t1 = _to_lpips_tensor_from_bgr(img_b_bgr, torch).to(device)

        with torch.no_grad():
            value = loss_fn(t0, t1).item()

        return float(value), "ok"
    except Exception as exc:
        return None, f"lpips failed: {exc}"


def _compute_quality_metrics(reference_bgr, reconstructed_bgr, splat_count):
    if reference_bgr.shape != reconstructed_bgr.shape:
        raise ValueError(
            "Metric computation requires same image shape. "
            f"Got {reference_bgr.shape} vs {reconstructed_bgr.shape}."
        )

    psnr = float(peak_signal_noise_ratio(reference_bgr, reconstructed_bgr, data_range=255))
    ssim = _compute_ssim_bgr(reference_bgr, reconstructed_bgr)
    lpips_value, lpips_status = _compute_lpips_if_available(reference_bgr, reconstructed_bgr)

    if splat_count > 0:
        psnr_per_splat = psnr / float(splat_count)
    else:
        psnr_per_splat = 0.0

    psnr_per_splat_scaled = False
    if psnr_per_splat < 1.0:
        psnr_per_splat *= 1000.0
        psnr_per_splat_scaled = True

    return {
        "splat_count": int(splat_count),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "lpips_status": lpips_status,
        "psnr_per_splat": float(psnr_per_splat),
        "psnr_per_splat_scaled_x1000": psnr_per_splat_scaled,
    }


def _parse_selected_trees(raw_values):
    """
    Parse CLI values for --trees, supporting both space-separated and comma-separated input.
    """
    if not raw_values:
        return set(VALID_TREE_NAMES)

    tokens = []
    for value in raw_values:
        parts = [item.strip().lower() for item in value.split(",") if item.strip()]
        tokens.extend(parts)

    if not tokens:
        return set(VALID_TREE_NAMES)

    invalid = sorted([name for name in tokens if name not in VALID_TREE_NAMES])
    if invalid:
        raise ValueError(
            f"Invalid tree name(s): {invalid}. Valid options: {sorted(VALID_TREE_NAMES)}"
        )

    return set(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-splat sweeps across tree structures."
    )
    parser.add_argument(
        "--trees",
        nargs="+",
        default=None,
        help=(
            "Tree(s) to run. Supports space or comma separated names. "
            "Valid: quadtree, kdtree, bsptree, upright_kdtree"
        ),
    )
    args = parser.parse_args()

    try:
        selected_trees = _parse_selected_trees(args.trees)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Selected trees: {sorted(selected_trees)}")

    color_path = "data/img.jpg"
    normal_path = "data/visual_normal.png"
    depth_path = "data/visual_depth.png"
    
    if os.path.exists(color_path) and os.path.exists(normal_path) and os.path.exists(depth_path):
        print("Testing single Gaussian splat per leaf across quadtree, kd-tree, and bsp-tree splitting variants...")
        
        # Load Color Image
        img_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        
        # Load Normal Image
        normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)

        # Load depth image (raw values preserved)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        color_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Configuration Lists to iterate
        min_sizes_to_test = [2, 4, 8, 16, 32]
        rgb_thresholds_to_test = [(30, 90), (50, 150), (100, 200)]

        tree_experiments = [
            (
                "quadtree",
                Quadtree,
                lambda min_size, dot_threshold, rgb_t1, rgb_t2: {
                    'color_image': img_bgr,
                    'normal_image': normal_bgr,
                    'depth_image': depth_raw,
                    'rgb_threshold1': rgb_t1,
                    'rgb_threshold2': rgb_t2,
                    'min_size': min_size,
                    'dot_threshold': dot_threshold,
                },
            ),
            (
                "kdtree",
                KdTree,
                lambda min_size, dot_threshold, rgb_t1, rgb_t2: {
                    'color_image': img_bgr,
                    'normal_image': normal_bgr,
                    'depth_image': depth_raw,
                    'rgb_threshold1': rgb_t1,
                    'rgb_threshold2': rgb_t2,
                    'min_size': min_size,
                    'dot_threshold': dot_threshold,
                },
            ),
            (
                "upright_kdtree",
                UprightKdTree,
                lambda min_size, dot_threshold, rgb_t1, rgb_t2: {
                    'color_image': img_bgr,
                    'normal_image': normal_bgr,
                    'depth_image': depth_raw,
                    'rgb_threshold1': rgb_t1,
                    'rgb_threshold2': rgb_t2,
                    'min_size': min_size,
                    'dot_threshold': dot_threshold,
                },
            ),
            (
                "bsptree",
                BSPTree,
                lambda min_size, dot_threshold, rgb_t1, rgb_t2: {
                    'color_image': img_bgr,
                    'normal_image': normal_bgr,
                    'depth_image': depth_raw,
                    'rgb_threshold1': rgb_t1,
                    'rgb_threshold2': rgb_t2,
                    'min_area': min_size * min_size,
                    'dot_threshold': dot_threshold,
                    'max_depth': 15,
                },
            ),
        ]

        tree_experiments = [
            experiment for experiment in tree_experiments if experiment[0] in selected_trees
        ]
        
        for min_size in min_sizes_to_test:
            for rgb_t1, rgb_t2 in rgb_thresholds_to_test:
                print(f"\n=======================================================")
                print(f"Testing Min Size: {min_size} | RGB Thresholds: {rgb_t1}-{rgb_t2}")
                print(f"=======================================================")

                for tree_name, tree_cls, kwargs_builder in tree_experiments:
                    out_dir = f"results/{tree_name}_min_size-{min_size}_threshold_{rgb_t1}-{rgb_t2}"
                    os.makedirs(out_dir, exist_ok=True)

                    print(f"  -> Tree: {tree_name}")

                    combined_edge_path = os.path.join(
                        out_dir,
                        f"{tree_name}_combined_rgb_depth_edges.png",
                    )
                
                    box_color = (0, 255, 0)
                    tree = tree_cls(**kwargs_builder(min_size, 0.99, rgb_t1, rgb_t2))

                    rendered_splats, splat_count = tree.render_splats_to_image()
                    rgb_with_boxes = tree.draw(tree.color_image, color=box_color, thickness=1)
                    splats_with_boxes = tree.draw(rendered_splats, color=box_color, thickness=1)

                    if tree_name == "upright_kdtree":
                        rgb_with_boxes = tree.rotate_back_and_refill(rgb_with_boxes)
                        splats_with_boxes = tree.rotate_back_and_refill(splats_with_boxes)
                        rendered_splats = tree.rotate_back_and_refill(rendered_splats)

                    output_rgb_with_boxes = rgb_with_boxes
                    output_splats_with_boxes = splats_with_boxes
                    output_rendered_splats = rendered_splats

                    cv2.imwrite(combined_edge_path, tree.combined_edges_map)
                    cv2.imwrite(os.path.join(out_dir, f"{tree_name}_rgb.png"), output_rgb_with_boxes)
                    cv2.imwrite(os.path.join(out_dir, f"{tree_name}_splats_with_boxes.png"), output_splats_with_boxes)
                    cv2.imwrite(os.path.join(out_dir, f"{tree_name}_splats_clean.png"), output_rendered_splats)

                    metrics = _compute_quality_metrics(
                        reference_bgr=img_bgr,
                        reconstructed_bgr=output_rendered_splats,
                        splat_count=splat_count,
                    )

                    json_path = os.path.join(out_dir, "single_splat_quality_metrics.json")
                    with open(json_path, "w") as f:
                        json.dump(metrics, f, indent=4)

        print("\nAll tree permutations correctly tested and dumped flawlessly into their specific structured 'results/' sub-directories!")
    else:
        print("Required data files not found in 'data/' to run the test block comprehensively. We need depth, color, and normal.")
