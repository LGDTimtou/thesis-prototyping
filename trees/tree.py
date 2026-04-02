import numpy as np
import cv2

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
    ):

        if color_image is None or normal_image is None or depth_image is None:
            raise ValueError(
                f"{self.__class__.__name__} requires color_image, normal_image, and depth_image."
            )
        
        self.color_image = color_image

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
        self.edge_map = maps["combined_edges_map"]
        self.rgb_edges_map = maps["rgb_edges_map"]
        self.depth_edges_map = maps["depth_edges_map"]
        self.combined_edges_map = maps["combined_edges_map"]
        self.min_size = min_size
        self.dot_threshold = dot_threshold

        self.h, self.w = self.normal_map.shape[:2]

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
        if self.normal_map is not None:
            patch_normals = self.normal_map[y:y + h, x:x + w]

            if patch_normals.size > 0:
                mean_normal = np.mean(patch_normals, axis=(0, 1))
                norm_len = np.linalg.norm(mean_normal)

                if norm_len > 0:
                    mean_normal /= norm_len

                dots = np.sum(patch_normals * mean_normal, axis=2)
                if np.mean(dots) < self.dot_threshold:
                    return True

        if self.edge_map is not None:
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

