import h5py
import numpy as np
import cv2

def convert_hdf5_to_images(depth_h5_path, normal_h5_path=None):
    import os
    
    # 1. Load Depth (Meters)
    print(f"Loading '{depth_h5_path}'...")
    with h5py.File(depth_h5_path, 'r') as f:
        depth_raw = np.array(f['dataset'])
    
    # --- PROCESS DEPTH ---
    # Real HDF5 depth maps often contain NaNs and Infs for background/sky. We must handle them safely.
    valid_mask = np.isfinite(depth_raw)
    
    # Calculate an optimal max distance dynamically to get a good visual mapping (e.g., 99th percentile)
    if np.any(valid_mask):
        max_dist = np.percentile(depth_raw[valid_mask], 99)
        min_dist = np.min(depth_raw[valid_mask])
    else:
        max_dist = 20.0
        min_dist = 0.0

    print(f"Depth mapped dynamically from {min_dist:.2f}m to {max_dist:.2f}m")

    # Deep copy the array and manually lock all broken abstracts (NaN, Inf) exactly to max distance natively
    depth_clean = np.array(depth_raw, dtype=np.float32)
    depth_clean[~valid_mask] = max_dist
    
    # Clip mathematically exactly to the bounds before running any multiplication/division
    depth_clipped = np.clip(depth_clean, min_dist, max_dist)
    
    # Normalize cleanly between 0.0 and 1.0 (with safe +1e-8 scalar to avoid DivideByZero mathematically)
    depth_normalized = (depth_clipped - min_dist) / ((max_dist - min_dist) + 1e-8)

    # Convert to standard 16-bit format to retain computational precision (0-65535)
    depth_16bit = (depth_normalized * 65535.0).astype(np.uint16)
    cv2.imwrite("data/visual_depth_16bit.png", depth_16bit)

    # Convert to standard 8-bit viewable image strictly for the colormap
    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    
    # Apply a colormap (INFERNO or JET) to make it easy to see depth changes visually!
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite("data/visual_depth_colormap.png", depth_colored)
    
    print("Depth Images correctly generated: 'data/visual_depth_16bit.png' (for precision) and 'data/visual_depth_colormap.png' (for visual inspection)")

    # --- PROCESS NORMALS (Optional) ---
    if normal_h5_path and os.path.exists(normal_h5_path):
        print(f"Loading '{normal_h5_path}'...")
        with h5py.File(normal_h5_path, 'r') as f:
            normals_raw = np.array(f['dataset'])
            
        # Map [-1, 1] range to [0, 255] safely
        normals_vis = ((normals_raw + 1.0) * 0.5 * 255).astype(np.uint8)
        normals_bgr = cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite("data/visual_normal.png", normals_bgr)
        print("Normal Image correctly generated: 'data/visual_normal.png'")
    else:
        print("No normal HDF5 path provided or file doesn't exist, skipping normal map generation.")

# Run it
convert_hdf5_to_images('data/frame.0000.depth_meters.hdf5', None)