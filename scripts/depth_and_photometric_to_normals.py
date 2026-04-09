import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


def remove_quantization_artifacts(depth: np.ndarray) -> np.ndarray:
    """
    Reduce quantization contours using morphological and interpolation techniques.
    """
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    
    # Morphological operations to connect quantization gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(depth_uint8, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    # Use inpainting-like smoothing to break harsh transitions
    # Apply small Gaussian then bilateral to smooth while keeping structure
    smooth1 = cv2.GaussianBlur(processed.astype(np.float32), (3, 3), 0.5)
    smooth2 = cv2.bilateralFilter(smooth1.astype(np.uint8), 5, 20, 20).astype(np.float32) / 255.0
    
    return smooth2


def apply_bilateral_filter(depth: np.ndarray, d: int = 9, sigma_color: float = 0.1, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filtering to preserve edges while smoothing."""
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(depth_uint8, d, sigma_color * 255, sigma_space)
    return filtered.astype(np.float32) / 255.0


def compute_normals_from_depth(
    depth: np.ndarray, smooth_sigma: float = 1.0, use_bilateral: bool = True
) -> np.ndarray:
    """
    Compute smooth surface normals from a depth map using filtered gradients.
    Aggressive preprocessing to remove quantization/JPEG artifacts.

    Args:
        depth: Grayscale depth map (H x W) as float in [0, 1].
        smooth_sigma: Smoothing strength (Gaussian sigma or bilateral iterations).
        use_bilateral: Use bilateral filtering (better edge preservation) vs Gaussian.

    Returns:
        Normal map (H x W x 3) with normals in [-1, 1] range and unit length.
    """
    # Ensure depth is float
    depth = depth.astype(np.float32)

    # Normalize depth to [0, 1]
    if depth.max() > 1.0:
        depth = depth / 255.0

    # Step 1: Remove quantization/JPEG artifacts
    print("  - Removing quantization artifacts...")
    depth = remove_quantization_artifacts(depth)
    depth = np.clip(depth, 0, 1)

    # Step 2: Upsample depth to reduce quantization effects
    print("  - Upsampling depth...")
    h, w = depth.shape
    depth_upsampled = cv2.resize(depth, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Step 3: Apply strong smoothing on upsampled depth
    if smooth_sigma > 0:
        if use_bilateral:
            # Bilateral: preserves edges, removes noise
            sigma_space = max(5, smooth_sigma * 15)
            depth_smooth = apply_bilateral_filter(depth_upsampled, d=11, sigma_color=0.08, sigma_space=sigma_space)
        else:
            # Gaussian + median
            depth_smooth = gaussian_filter(depth_upsampled, sigma=smooth_sigma * 2)
            depth_smooth = cv2.medianBlur((depth_smooth * 255).astype(np.uint8), 7).astype(np.float32) / 255.0
    else:
        depth_smooth = depth_upsampled

    # Step 4: Compute gradients on smooth upsampled depth
    grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=7)

    # Initialize normal map
    h, w = depth_smooth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    # Normal vector formula: N = [-grad_x, -grad_y, 1]
    # (assumes positive Z points away from surface)
    normals[:, :, 0] = -grad_x
    normals[:, :, 1] = -grad_y
    normals[:, :, 2] = 1.0

    # Normalize to unit vectors
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals = normals / norms

    # Step 5: Downsample normals back to original resolution
    normals_downsampled = cv2.resize(normals, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    
    # Renormalize after downsampling
    norms = np.linalg.norm(normals_downsampled, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals_downsampled / norms

    return normals


def blend_normals_with_photometric(
    normals: np.ndarray, photometric: np.ndarray, blend_weight: float = 0.1
) -> np.ndarray:
    """
    Optionally blend computed normals with photometric cues.
    (Simple approach: use photometric image to enhance high-frequency details.)

    Args:
        normals: Computed normal map (H x W x 3) in [-1, 1].
        photometric: RGB photometric image (H x W x 3) in [0, 255].
        blend_weight: Weight for photometric guidance (0 = no blending).

    Returns:
        Blended normal map (H x W x 3) in [-1, 1].
    """
    if blend_weight == 0 or photometric is None:
        return normals

    # Convert photometric to grayscale for guidance
    photo_gray = cv2.cvtColor(photometric, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Compute high-frequency details in photometric image using Laplacian
    laplacian = cv2.Laplacian(photo_gray, cv2.CV_32F)
    laplacian = np.clip(laplacian / laplacian.max(), -1, 1) if laplacian.max() > 0 else laplacian

    # Blend: enhance detail in normal's Z component
    blended = normals.copy()
    blended[:, :, 2] += blend_weight * laplacian

    # Renormalize to unit vectors
    norms = np.linalg.norm(blended, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    blended = blended / norms

    return blended


def encode_normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """
    Encode normal vectors ([-1, 1] range) to RGB image ([0, 255] range).
    Maps X, Y, Z components to R, G, B channels.
    """
    # Map from [-1, 1] to [0, 255]
    encoded = ((normals + 1.0) * 0.5 * 255).astype(np.uint8)
    return encoded


def compute_and_save_normals(
    depth_path: str,
    photometric_path: str = None,
    output_path: str = None,
    smooth_sigma: float = 2.0,
    blend_weight: float = 0.0,
    use_bilateral: bool = True,
) -> None:
    """
    Main function to compute normals from depth and optional photometric image.

    Args:
        depth_path: Path to black-and-white depth image.
        photometric_path: Optional path to RGB photometric image.
        output_path: Path for output normal map.
        smooth_sigma: Smoothing strength (higher = smoother). Reduced from 3.0 due to upsampling strategy.
        blend_weight: Weight for photometric guidance blending.
        use_bilateral: Use bilateral filtering (better edge preservation).
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    # Load depth image
    print(f"Loading depth from: {depth_path}")
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {depth_path}")

    # Load photometric image if provided
    photometric = None
    if photometric_path and os.path.exists(photometric_path):
        print(f"Loading photometric from: {photometric_path}")
        photometric = cv2.imread(photometric_path, cv2.IMREAD_COLOR)
        if photometric is None:
            print(f"Warning: Failed to read photometric image: {photometric_path}")
        elif photometric.shape[:2] != depth.shape:
            print(
                f"Warning: Photometric shape {photometric.shape} != depth shape {depth.shape}. Skipping blend."
            )
            photometric = None

    # Compute normals
    filter_type = "bilateral" if use_bilateral else "gaussian+median"
    print(f"Computing normals with smooth_sigma={smooth_sigma} ({filter_type})...")
    print(f"  Input depth shape: {depth.shape}")
    normals = compute_normals_from_depth(depth, smooth_sigma=smooth_sigma, use_bilateral=use_bilateral)
    print(f"  Output normal shape: {normals.shape}")

    # Optionally blend with photometric
    if photometric is not None and blend_weight > 0:
        print(f"Blending with photometric (weight={blend_weight})...")
        normals = blend_normals_with_photometric(normals, photometric, blend_weight)

    # Encode to RGB
    normal_rgb = encode_normals_to_rgb(normals)

    # Convert to BGR for OpenCV (normals already RGB)
    normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    # Save
    if output_path is None:
        output_path = "data/visual_normal_computed.png"

    if not cv2.imwrite(output_path, normal_bgr):
        raise IOError(f"Failed to write output image: {output_path}")

    print(f"Normal map saved to: {output_path}")


def pick_defaults() -> tuple:
    """Detect default input files if they exist."""
    depth_candidates = [
        "data/visual_depth_bw.jpg",
        "data/visual_depth_bw.png",
        "data/visual_depth.jpg",
        "data/visual_depth.png",
    ]
    photometric_candidates = [
        "data/visual_deth.png",
        "data/visual_depth.png",
        "data/img.jpg",
    ]

    depth_path = next((p for p in depth_candidates if os.path.exists(p)), depth_candidates[0])
    photometric_path = next(
        (p for p in photometric_candidates if os.path.exists(p)), None
    )

    return depth_path, photometric_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute smooth surface normals from a depth map and optional photometric image."
    )

    depth_default, photo_default = pick_defaults()

    parser.add_argument(
        "--depth",
        default=depth_default,
        help="Path to black-and-white depth image.",
    )
    parser.add_argument(
        "--photometric",
        default=photo_default,
        help="Optional path to RGB photometric image for guidance.",
    )
    parser.add_argument(
        "--output",
        default="data/visual_normal_computed.png",
        help="Path for output normal map.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=2.0,
        help="Smoothing strength (higher = smoother). Upsampling handles much of the smoothing. Default: 2.0",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.0,
        help="Photometric blending weight (0 = no blend, >0 = use photometric details).",
    )
    parser.add_argument(
        "--gaussian-only",
        action="store_true",
        help="Use Gaussian+median filtering instead of bilateral (bilateral is default).",
    )

    args = parser.parse_args()

    compute_and_save_normals(
        depth_path=args.depth,
        photometric_path=args.photometric,
        output_path=args.output,
        smooth_sigma=args.smooth,
        blend_weight=args.blend,
        use_bilateral=not args.gaussian_only,
    )


if __name__ == "__main__":
    main()
