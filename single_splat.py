import cv2
import numpy as np
import os
from quadtree import Quadtree

def extract_single_splat_per_square(quadtree, color_image):
    """
    Extracts exactly one Gaussian splat per leaf node in the quadtree.
    Each splat takes the RGB color of the middle pixel, uses that as the center,
    and scales to completely match the square.
    """
    splats = []
    
    def _traverse(node):
        if node.is_leaf:
            # Calculate the exact center of the square
            cx = node.x + node.width // 2
            cy = node.y + node.height // 2
            
            # Ensure the center is within the image boundaries
            cx = min(max(cx, 0), color_image.shape[1] - 1)
            cy = min(max(cy, 0), color_image.shape[0] - 1)
            
            # Take the RGB color of the middle pixel
            color = color_image[cy, cx]
            
            # Scale so it completely matches the square
            # width and height represent the full bounding box of the square
            splat = {
                'x': float(cx),
                'y': float(cy),
                'color': color.tolist(), # BGR list
                'scale_x': float(node.width),
                'scale_y': float(node.height),
                'square_x': node.x,
                'square_y': node.y
            }
            splats.append(splat)
        else:
            for child in node.children:
                _traverse(child)
                
    _traverse(quadtree.root)
    return splats


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
        print("Testing single Gaussian splat per square across 4 Quadtree splitting variants...")
        
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
        
        import json
        
        # Configuration Lists to iterate
        min_sizes_to_test = [2, 4, 8, 16, 32]
        rgb_thresholds_to_test = [(30, 90), (50, 150), (100, 200)]
        
        for min_size in min_sizes_to_test:
            for rgb_t1, rgb_t2 in rgb_thresholds_to_test:
                # --- Process RGB edges with current dynamically passed threshold arrays ---
                rgb_edges_mask = cv2.Canny(color_gray, threshold1=rgb_t1, threshold2=rgb_t2)
                
                # Mathematical Stack of both structural edge detection variants
                combined_edges_mask = cv2.bitwise_or(depth_edges_mask, rgb_edges_mask)
                
                # Output Directory specific exactly to this configured run
                out_dir = f"results/quadtree_min_size-{min_size}_threshold_{rgb_t1}-{rgb_t2}"
                os.makedirs(out_dir, exist_ok=True)
                
                print(f"\n=======================================================")
                print(f"Testing Min Size: {min_size} | RGB Thresholds: {rgb_t1}-{rgb_t2}")
                print(f"Directory: {out_dir}/")
                print(f"=======================================================")
                
                # Dictionary to hold splats iteration counts internally
                splats_counts = {}
                
                # Define 4 configuration tuples: (Name, Box Color, Quadtree args dict)
                configs = [
                    ("normals_only", (0, 0, 255), {'normal_map': normals_raw_vectors, 'edge_map': None, 'min_size': min_size, 'dot_threshold': 0.99}),
                    ("depth_edges_only", (255, 0, 0), {'normal_map': None, 'edge_map': depth_edges_mask, 'min_size': min_size}),
                    ("color_edges_only", (0, 255, 255), {'normal_map': None, 'edge_map': rgb_edges_mask, 'min_size': min_size}),
                    ("combined_all", (0, 255, 0), {'normal_map': normals_raw_vectors, 'edge_map': combined_edges_mask, 'min_size': min_size, 'dot_threshold': 0.99})
                ]
                
                for name, box_color, qt_kwargs in configs:
                    print(f"  -> Processing algorithm limit: {name}...")
                    qt = Quadtree(**qt_kwargs)
                    
                    splats = extract_single_splat_per_square(qt, img_bgr)
                    splats_counts[name] = len(splats)
                    
                    rendered_splats = render_splats_to_image(splats, img_bgr.shape)
                    
                    rgb_with_boxes = qt.draw(img_bgr, color=box_color, thickness=1)
                    splats_with_boxes = qt.draw(rendered_splats, color=box_color, thickness=1)
                    
                    # Save natively into the folder
                    cv2.imwrite(os.path.join(out_dir, f"quadtree_rgb_{name}.png"), rgb_with_boxes)
                    cv2.imwrite(os.path.join(out_dir, f"rendered_splats_with_boxes_{name}.png"), splats_with_boxes)
                    cv2.imwrite(os.path.join(out_dir, f"rendered_splats_clean_{name}.png"), rendered_splats)
                    
                # Structurally dump the output metrics into JSON specifically for this folder index
                json_path = os.path.join(out_dir, "splats_count.json")
                with open(json_path, "w") as f:
                    json.dump(splats_counts, f, indent=4)
                    
        print("\nAll algorithm permutations correctly tested and dumped flawlessly into their specific structured 'results/' sub-directories!")
    else:
        print("Required data files not found in 'data/' to run the test block comprehensively. We need depth, color, and normal.")
