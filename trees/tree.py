import numpy as np
import cv2
import os

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData = None
    PlyElement = None

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



class TreeNode:
    def __init__(self, level=0):
        self.level = level
        self.children = []
        self.is_leaf = True


class Tree:
    def __init__(
        self,
        color_image=None,
        normal_image=None,
        depth_image=None,
        min_size=8,
        dot_threshold=0.98,
        rgb_threshold1=30,
        rgb_threshold2=90,
        depth_threshold1=15,
        depth_threshold2=50,
        use_normal_variance=True,
        use_rgb_edges=True,
        use_depth_edges=True,
    ):

        if color_image is None or normal_image is None or depth_image is None:
            raise ValueError(
                f"{self.__class__.__name__} requires color_image, normal_image, and depth_image."
            )
        
        self.color_image = color_image
        self.depth_image = depth_image
        self.use_normal_variance = bool(use_normal_variance)
        self.use_rgb_edges = bool(use_rgb_edges)
        self.use_depth_edges = bool(use_depth_edges)

        maps = self.build_feature_maps(
            color_image=color_image,
            normal_image=normal_image,
            depth_image=depth_image,
            rgb_threshold1=rgb_threshold1,
            rgb_threshold2=rgb_threshold2,
            depth_threshold1=depth_threshold1,
            depth_threshold2=depth_threshold2,
        )

        self.normal_map = maps["normal_map"]
        self.rgb_edges_map = maps["rgb_edges_map"]
        self.depth_edges_map = maps["depth_edges_map"]

        selected_edge_maps = []
        if self.use_rgb_edges:
            selected_edge_maps.append(self.rgb_edges_map)
        if self.use_depth_edges:
            selected_edge_maps.append(self.depth_edges_map)

        if selected_edge_maps:
            edge_map = selected_edge_maps[0].copy()
            for extra_map in selected_edge_maps[1:]:
                edge_map = cv2.bitwise_or(edge_map, extra_map)
        else:
            edge_map = np.zeros_like(self.rgb_edges_map)

        self.edge_map = edge_map
        self.combined_edges_map = edge_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold

        self.h, self.w = self.normal_map.shape[:2]

    @staticmethod
    def _depth_to_float(depth_image):
        if depth_image is None:
            return None

        if len(depth_image.shape) == 2:
            return depth_image.astype(np.float32)

        return cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def _safe_normalize(vector, eps=1e-8):
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _resolve_intrinsics(image_width, image_height, fx=None, fy=None, cx=None, cy=None):
        if fx is None or fx <= 0:
            fx = float(max(image_width, image_height))
        if fy is None or fy <= 0:
            fy = float(max(image_width, image_height))
        if cx is None:
            cx = (image_width - 1) * 0.5
        if cy is None:
            cy = (image_height - 1) * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _project_depth_to_3d_map(depth_map, fx, fy, cx, cy, depth_scale=1.0):
        if not hasattr(cv2, "rgbd") or not hasattr(cv2.rgbd, "depthTo3d"):
            raise RuntimeError(
                "cv2.rgbd.depthTo3d is unavailable. Install opencv-contrib-python."
            )

        depth_scale = float(depth_scale)
        if depth_scale <= 0:
            raise ValueError("depth_scale must be > 0.")

        depth_metric = depth_map.astype(np.float32) / depth_scale
        depth_metric[~np.isfinite(depth_metric)] = 0.0

        k_matrix = np.array(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return cv2.rgbd.depthTo3d(depth_metric, k_matrix)

    @staticmethod
    def _infer_depth_scale(depth_image):
        return 1.0

    @classmethod
    def _quat_from_z_to_normal(cls, normal_vec):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = cls._safe_normalize(np.asarray(normal_vec, dtype=np.float32))

        dot = float(np.clip(np.dot(z_axis, n), -1.0, 1.0))
        if dot >= 1.0 - 1e-7:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if dot <= -1.0 + 1e-7:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        axis = np.cross(z_axis, n)
        axis = cls._safe_normalize(axis)
        angle = float(np.arccos(dot))
        half = 0.5 * angle
        sin_half = float(np.sin(half))
        cos_half = float(np.cos(half))
        return np.array([
            cos_half,
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        ], dtype=np.float32)



    @staticmethod
    def _normal_to_vectors(normal_image):
        if normal_image is None:
            return None

        if normal_image.ndim == 3 and normal_image.shape[2] >= 3:
            if normal_image.dtype == np.uint8:
                rgb = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
                return (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0

            normal_float = normal_image.astype(np.float32)
            if np.min(normal_float) >= -1.0 and np.max(normal_float) <= 1.0:
                return normal_float

            if np.max(normal_float) > 1.0:
                return (normal_float / 255.0) * 2.0 - 1.0

        raise ValueError("normal_image must be HxWx3 uint8 (BGR) or float vectors in [-1, 1].")

    @staticmethod
    def _depth_to_u8(depth_image):
        if depth_image is None:
            return None

        if len(depth_image.shape) == 2:
            if depth_image.dtype == np.uint8:
                return depth_image

            depth_float = depth_image.astype(np.float32)
            return (depth_float / (np.max(depth_float) + 1e-8) * 255.0).astype(np.uint8)

        return cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    @classmethod
    def build_feature_maps(
        cls,
        color_image,
        normal_image=None,
        depth_image=None,
        rgb_threshold1=30,
        rgb_threshold2=90,
        depth_threshold1=15,
        depth_threshold2=50,
    ):
        color_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        rgb_edges_map = cv2.Canny(color_gray, threshold1=rgb_threshold1, threshold2=rgb_threshold2)

        normal_map = cls._normal_to_vectors(normal_image)

        depth_u8 = cls._depth_to_u8(depth_image)
        if depth_u8 is None:
            depth_edges_map = np.zeros_like(rgb_edges_map)
        else:
            depth_edges_map = cv2.Canny(depth_u8, threshold1=depth_threshold1, threshold2=depth_threshold2)

        combined_edges_map = cv2.bitwise_or(depth_edges_map, rgb_edges_map)

        return {
            "normal_map": normal_map,
            "rgb_edges_map": rgb_edges_map,
            "depth_edges_map": depth_edges_map,
            "combined_edges_map": combined_edges_map,
        }

    def _should_split(self, x, y, w, h):
        if self.use_normal_variance and self.normal_map is not None:
            patch_normals = self.normal_map[y:y + h, x:x + w]

            if patch_normals.size > 0:
                mean_normal = np.mean(patch_normals, axis=(0, 1))
                norm_len = np.linalg.norm(mean_normal)

                if norm_len > 0:
                    mean_normal /= norm_len

                dots = np.sum(patch_normals * mean_normal, axis=2)
                if np.mean(dots) < self.dot_threshold:
                    return True

        if (self.use_rgb_edges or self.use_depth_edges) and self.edge_map is not None:
            patch_edges = self.edge_map[y:y + h, x:x + w]
            if np.any(patch_edges > 0):
                return True

        return False


    def extract_single_splat_per_leaf(self):
        """Extract exactly one Gaussian splat per leaf node in any supported tree."""
        splats = []

        def _traverse(node):
            if node.is_leaf:
                x, y, width, height, cx, cy = _node_geometry(node)
                cx = min(max(int(round(cx)), 0), self.color_image.shape[1] - 1)
                cy = min(max(int(round(cy)), 0), self.color_image.shape[0] - 1)
                color = self.color_image[cy, cx]
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

        _traverse(self.root)
        return splats

    def extract_single_3d_splat_per_leaf(
        self,
        depth_thickness_ratio=0.05,
        min_depth_thickness=1e-4,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        opacity=1.0,
        depth_scale=None,
    ):
        """Extract one 3D Gaussian-like splat per leaf and attach orientation/SH DC color."""
        sh_c0 = 0.28209479177387814

        def _opacity_to_logit(alpha):
            a = float(alpha)
            if a >= 1.0:
                return 20.0
            if a <= 0.0:
                return -20.0
            return float(np.log(a / (1.0 - a)))

        if self.normal_map is None:
            raise ValueError("normal_map is required to orient 3D splats.")

        depth_map = self._depth_to_float(self.depth_image)
        if depth_map is None:
            raise ValueError("depth_image is required to place 3D splats.")

        fx, fy, cx, cy = self._resolve_intrinsics(self.w, self.h, fx=fx, fy=fy, cx=cx, cy=cy)
        if depth_scale is None:
            depth_scale = 1.0
        points_3d = self._project_depth_to_3d_map(depth_map, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale)

        splats = []

        def _traverse(node):
            if node.is_leaf:
                x, y, width, height, center_x, center_y = _node_geometry(node)
                px = min(max(int(round(center_x)), 0), self.w - 1)
                py = min(max(int(round(center_y)), 0), self.h - 1)

                depth_value = float(depth_map[py, px])
                point_xyz = points_3d[py, px].astype(np.float32)
                is_valid_point = np.isfinite(point_xyz).all() and depth_value > 0.0 and float(point_xyz[2]) > 0.0
                if not is_valid_point:
                    return

                world_x = float(point_xyz[0])
                world_y = float(point_xyz[1])
                world_z = float(point_xyz[2])

                x0 = min(max(int(x), 0), self.w - 1)
                y0 = min(max(int(y), 0), self.h - 1)
                x1 = min(self.w, x0 + max(1, int(width)))
                y1 = min(self.h, y0 + max(1, int(height)))

                # Align splats parallel to image plane (identity quaternion)
                # This keeps splats aligned with camera axes: x=right, y=down, z=forward
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

                square_world_u = max(1.0, float(width)) * max(world_z, 1e-8) / float(fx)
                square_world_v = max(1.0, float(height)) * max(world_z, 1e-8) / float(fy)

                # Scale divisor based on square size: smaller squares get slightly more downscaling
                avg_square_size = (width + height) / 2.0
                divisor = 1.0 + 1.0 / (1.0 + avg_square_size / 10.0)
                
                scale_u = square_world_u / divisor
                scale_v = square_world_v / divisor

                scale_n = max(
                    float(min_depth_thickness),
                    min(scale_u, scale_v) * float(depth_thickness_ratio),
                )

                center_bgr = self.color_image[py, px].astype(np.float32)
                center_rgb = center_bgr[::-1] / 255.0

                splats.append({
                    "x": float(world_x),
                    "y": float(world_y),
                    "z": float(world_z),
                    "f_dc_0": float((center_rgb[0] - 0.5) / sh_c0),
                    "f_dc_1": float((center_rgb[1] - 0.5) / sh_c0),
                    "f_dc_2": float((center_rgb[2] - 0.5) / sh_c0),
                    "opacity": _opacity_to_logit(opacity),
                    "scale_0": float(np.log(scale_u + 1e-8)),
                    "scale_1": float(np.log(scale_v + 1e-8)),
                    "scale_2": float(np.log(scale_n + 1e-8)),
                    "rot_0": float(quat_wxyz[0]),
                    "rot_1": float(quat_wxyz[1]),
                    "rot_2": float(quat_wxyz[2]),
                    "rot_3": float(quat_wxyz[3]),
                })
            else:
                for child in node.children:
                    _traverse(child)

        _traverse(self.root)
        return splats

    def export_3d_splats_to_ply(
        self,
        output_ply_path,
        depth_thickness_ratio=0.05,
        min_depth_thickness=1e-4,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        opacity=1.0,
        depth_scale=None,
    ):
        """Export one 3D splat per leaf to a PLY file using plyfile."""
        if PlyData is None or PlyElement is None:
            raise ImportError("plyfile is required for PLY export. Install it with: pip install plyfile")

        splats = self.extract_single_3d_splat_per_leaf(
            depth_thickness_ratio=depth_thickness_ratio,
            min_depth_thickness=min_depth_thickness,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            opacity=opacity,
            depth_scale=depth_scale,
        )

        out_dir = os.path.dirname(output_ply_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        vertex_dtype = np.dtype([
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ])

        vertices = np.empty(len(splats), dtype=vertex_dtype)
        for idx, s in enumerate(splats):
            vertices[idx] = (
                s["x"],
                s["y"],
                s["z"],
                s["f_dc_0"],
                s["f_dc_1"],
                s["f_dc_2"],
                s["opacity"],
                s["scale_0"],
                s["scale_1"],
                s["scale_2"],
                s["rot_0"],
                s["rot_1"],
                s["rot_2"],
                s["rot_3"],
            )

        ply_data = PlyData([PlyElement.describe(vertices, "vertex")], text=False)
        ply_data.write(output_ply_path)

        return len(splats)


    def render_splats_to_image(self):
        """
        Utility to render the extracted Gaussian splats back into an image
        to visually verify that they cover the squares and match colors.
        """

        splats = self.extract_single_splat_per_leaf()


        # Create an empty floating point canvas
        rendered = np.zeros((self.color_image.shape[0], self.color_image.shape[1], 3), dtype=np.float32)
        weight_sum = np.zeros((self.color_image.shape[0], self.color_image.shape[1], 1), dtype=np.float32)

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
            y_max = min(self.color_image.shape[0], int(cy) + box_h * 2 + 1)
            x_min = max(0, int(cx) - box_w * 2)
            x_max = min(self.color_image.shape[1], int(cx) + box_w * 2 + 1)
            
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
        
        return np.clip(rendered, 0, 255).astype(np.uint8), len(splats)

