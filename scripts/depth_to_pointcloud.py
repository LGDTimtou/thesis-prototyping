import os
import cv2
import numpy as np


def depth_to_pointcloud_xyz(
    depth_path,
    output_xyz_path,
    color_path=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    depth_scale=None,
    stride=2,
    max_depth_m=None,
):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth_scale = 1.0 if depth_scale is None else float(depth_scale)
    if depth_scale <= 0:
        raise ValueError("depth_scale must be > 0")

    depth = depth.astype(np.float32) / depth_scale
    depth[~np.isfinite(depth)] = 0.0

    h, w = depth.shape[:2]
    fx = float(max(w, h) if fx is None or fx <= 0 else fx)
    fy = float(max(w, h) if fy is None or fy <= 0 else fy)
    cx = float((w - 1) * 0.5 if cx is None else cx)
    cy = float((h - 1) * 0.5 if cy is None else cy)

    if not hasattr(cv2, "rgbd") or not hasattr(cv2.rgbd, "depthTo3d"):
        raise RuntimeError("cv2.rgbd.depthTo3d is unavailable. Install opencv-contrib-python.")

    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    points_3d = cv2.rgbd.depthTo3d(depth, k)

    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR) if color_path else None
    if color_bgr is not None and color_bgr.shape[:2] != (h, w):
        raise ValueError(
            "Color and depth image sizes must match. "
            f"Got color={color_bgr.shape[:2]} depth={(h, w)}"
        )

    stride = max(1, int(stride))
    sampled_xyz = points_3d[::stride, ::stride].reshape(-1, 3)
    sampled_depth = depth[::stride, ::stride].reshape(-1)

    valid = np.isfinite(sampled_xyz).all(axis=1) & (sampled_depth > 0.0)
    if max_depth_m is not None:
        valid &= sampled_depth <= float(max_depth_m)

    xyz = sampled_xyz[valid].astype(np.float32)

    if color_bgr is not None:
        sampled_bgr = color_bgr[::stride, ::stride].reshape(-1, 3)[valid]
        rgb = sampled_bgr[:, ::-1].astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    data = np.hstack([xyz, rgb.astype(np.float32)])

    os.makedirs(os.path.dirname(output_xyz_path) or ".", exist_ok=True)
    np.savetxt(
        output_xyz_path,
        data,
        fmt="%.6f %.6f %.6f %d %d %d",
        header="x y z r g b",
        comments="# ",
    )

    return int(data.shape[0])


def run_depth_pointcloud_test():
    depth_path = "data/visual_depth.png"
    color_path = "data/img.jpg"
    output_xyz_path = "results/depth_pointcloud_test.xyz"

    if not os.path.exists(color_path):
        fallback_color = "data/frame.0000.color.jpg"
        if os.path.exists(fallback_color):
            color_path = fallback_color
        else:
            color_path = None

    num_points = depth_to_pointcloud_xyz(
        depth_path=depth_path,
        output_xyz_path=output_xyz_path,
        color_path=color_path,
        depth_scale=None,
        stride=2,
        max_depth_m=None,
    )

    print(f"Saved point cloud with {num_points} points to: {output_xyz_path}")


if __name__ == "__main__":
    run_depth_pointcloud_test()
