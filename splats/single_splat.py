import cv2
import numpy as np
import os
import json

from trees.quadtree import Quadtree
from trees.kd_tree import KdTree
from trees.bsp_tree import BSPTree


def _node_geometry(node):
    if hasattr(node, "polygon"):
        pts = np.array(node.polygon, dtype=np.float32)
        if pts.size == 0:
            return 0, 0, 1, 1, 0.0, 0.0

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        width = max(1, int(round(x_max - x_min)))
        height = max(1, int(round(y_max - y_min)))
        return int(round(x_min)), int(round(y_min)), width, height, cx, cy

    x = int(node.x)
    y = int(node.y)
    width = int(node.width)
    height = int(node.height)
    cx = x + width / 2.0
    cy = y + height / 2.0
    return x, y, width, height, cx, cy


def extract_single_splat_per_leaf(tree, color_image):
    """Extract exactly one Gaussian splat per leaf node in any supported tree."""
    splats = []

    def _traverse(node):
        if node.is_leaf:
            x, y, width, height, cx, cy = _node_geometry(node)
            cx = min(max(int(round(cx)), 0), color_image.shape[1] - 1)
            cy = min(max(int(round(cy)), 0), color_image.shape[0] - 1)
            color = color_image[cy, cx]
            splat = {
                'x': float(cx),
                'y': float(cy),
                'color': color.tolist(),
                'scale_x': float(width),
                'scale_y': float(height),
                'square_x': x,
                'square_y': y
            }
            splats.append(splat)
        else:
            for child in node.children:
                _traverse(child)

    _traverse(tree.root)
    return splats


def extract_single_splat_per_square(tree, color_image):
    return extract_single_splat_per_leaf(tree, color_image)


def render_splats_to_image(splats, img_shape):
    """
    Utility to render the extracted Gaussian splats back into an image
    to visually verify that they cover the squares and match colors.
    """
    # Create an empty floating point canvas
    rendered = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float32)
    weight_sum = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)

    for s in splats:
        cx, cy = s['x'], s['y']
        
        # Scale parameters so it completely matches the square
        # We can interpret "completely match" as the Gaussian extending 3 sigma to the edges
        sigma_x = s['scale_x'] / 6.0 + 1e-5
        sigma_y = s['scale_y'] / 6.0 + 1e-5
        
        color = np.array(s['color'], dtype=np.float32)
        
        # Bounding box for rendering (3 sigma is approximately half the scale)
        box_w = int(s['scale_x'] / 2) + 1
        box_h = int(s['scale_y'] / 2) + 1
        
        y_min = max(0, int(cy) - box_h * 2) # extend a bit to smooth out
        y_max = min(img_shape[0], int(cy) + box_h * 2 + 1)
        x_min = max(0, int(cx) - box_w * 2)
        x_max = min(img_shape[1], int(cx) + box_w * 2 + 1)
        
        if y_max <= y_min or x_max <= x_min:
            continue
            
        # Create a grid for the bounding box
        grid_y, grid_x = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Evaluate 2D Gaussian
        dx = grid_x - cx
        dy = grid_y - cy
        
        # Gaussian formula
        gaussian = np.exp(-((dx**2) / (2 * sigma_x**2) + (dy**2) / (2 * sigma_y**2)))
        gaussian = gaussian[..., np.newaxis] # Broadcast over color channels
        
        rendered[y_min:y_max, x_min:x_max] += color * gaussian
        weight_sum[y_min:y_max, x_min:x_max] += gaussian

    # Normalize colors by the accumulated weights to act as an alpha blend
    valid_mask_2d = weight_sum[..., 0] > 1e-5
    rendered[valid_mask_2d] = rendered[valid_mask_2d] / weight_sum[valid_mask_2d]
    
    # Fill any empty pixels with black
    rendered[~valid_mask_2d] = 0
    
    return np.clip(rendered, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    color_path = "data/img.jpg"
    if not os.path.exists(color_path):
        color_path = "data/frame.0000.color.jpg"
        
    normal_path = "data/visual_normal.png"
    depth_path = "data/visual_depth.png"
    
    if os.path.exists(color_path) and os.path.exists(normal_path) and os.path.exists(depth_path):
        print("Testing single Gaussian splat per leaf across quadtree, kd-tree, and bsp-tree splitting variants...")
        
        # Load Color Image
        img_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        
        # Load Normal Image
        normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
        normals_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
        normals_raw_vectors = (normals_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # Load and Process Depth edges
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_float = depth_raw.astype(np.float32)

        if len(depth_raw.shape) == 2:
            if depth_raw.dtype == np.uint8:
                depth_8u = depth_raw
            else:
                depth_8u = (depth_float / (np.max(depth_float) + 1e-8) * 255.0).astype(np.uint8)
        else:
            # Assuming it is already safely 3 channel
            depth_8u = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
            
        # --- Execute Pre-Calculated Canny Extractor on DEPTH ---
        depth_edges_mask = cv2.Canny(depth_8u, threshold1=15, threshold2=50)
        color_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Configuration Lists to iterate
        min_sizes_to_test = [2, 4, 8, 16, 32]
        rgb_thresholds_to_test = [(30, 90), (50, 150), (100, 200)]

        tree_experiments = [
            (
                "quadtree",
                Quadtree,
                lambda min_size, normal_map, edge_map, dot_threshold: {
                    'normal_map': normal_map,
                    'edge_map': edge_map,
                    'min_size': min_size,
                    'dot_threshold': dot_threshold,
                },
            ),
            (
                "kdtree",
                KdTree,
                lambda min_size, normal_map, edge_map, dot_threshold: {
                    'normal_map': normal_map,
                    'edge_map': edge_map,
                    'min_size': min_size,
                    'dot_threshold': dot_threshold,
                },
            ),
            (
                "bsptree",
                BSPTree,
                lambda min_size, normal_map, edge_map, dot_threshold: {
                    'normal_map': normal_map,
                    'edge_map': edge_map,
                    'min_area': min_size * min_size,
                    'dot_threshold': dot_threshold,
                    'max_depth': 15,
                },
            ),
        ]
        
        for min_size in min_sizes_to_test:
            for rgb_t1, rgb_t2 in rgb_thresholds_to_test:
                # --- Process RGB edges with current dynamically passed threshold arrays ---
                rgb_edges_mask = cv2.Canny(color_gray, threshold1=rgb_t1, threshold2=rgb_t2)
                
                # Mathematical Stack of both structural edge detection variants
                combined_edges_mask = cv2.bitwise_or(depth_edges_mask, rgb_edges_mask)
                
                print(f"\n=======================================================")
                print(f"Testing Min Size: {min_size} | RGB Thresholds: {rgb_t1}-{rgb_t2}")
                print(f"=======================================================")

                for tree_name, tree_cls, kwargs_builder in tree_experiments:
                    out_dir = f"results/{tree_name}_min_size-{min_size}_threshold_{rgb_t1}-{rgb_t2}"
                    os.makedirs(out_dir, exist_ok=True)

                    print(f"  -> Tree: {tree_name}")

                    tree_counts = {}
                    configs = [
                        ("normals_only", (0, 0, 255), kwargs_builder(min_size, normals_raw_vectors, None, 0.99)),
                        ("depth_edges_only", (255, 0, 0), kwargs_builder(min_size, None, depth_edges_mask, 0.98)),
                        ("color_edges_only", (0, 255, 255), kwargs_builder(min_size, None, rgb_edges_mask, 0.98)),
                        ("combined_all", (0, 255, 0), kwargs_builder(min_size, normals_raw_vectors, combined_edges_mask, 0.99)),
                    ]

                    for name, box_color, tree_kwargs in configs:
                        print(f"    -> Processing algorithm limit: {name}...")
                        tree = tree_cls(**tree_kwargs)

                        splats = extract_single_splat_per_leaf(tree, img_bgr)
                        tree_counts[name] = len(splats)

                        rendered_splats = render_splats_to_image(splats, img_bgr.shape)

                        rgb_with_boxes = tree.draw(img_bgr, color=box_color, thickness=1)
                        splats_with_boxes = tree.draw(rendered_splats, color=box_color, thickness=1)

                        cv2.imwrite(os.path.join(out_dir, f"{tree_name}_rgb_{name}.png"), rgb_with_boxes)
                        cv2.imwrite(os.path.join(out_dir, f"rendered_splats_with_boxes_{name}.png"), splats_with_boxes)
                        cv2.imwrite(os.path.join(out_dir, f"rendered_splats_clean_{name}.png"), rendered_splats)

                    json_path = os.path.join(out_dir, "splats_count.json")
                    with open(json_path, "w") as f:
                        json.dump(tree_counts, f, indent=4)
                    
        print("\nAll tree permutations correctly tested and dumped flawlessly into their specific structured 'results/' sub-directories!")
    else:
        print("Required data files not found in 'data/' to run the test block comprehensively. We need depth, color, and normal.")
