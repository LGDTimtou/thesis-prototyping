import cv2
import numpy as np
import os


class KdTreeNode:
    """A single axis-aligned region in the kd-tree."""

    def __init__(self, x, y, width, height, level=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.level = level
        self.children = []
        self.is_leaf = True

    def split(self, orientation):
        """Split the region into two children along the selected orientation."""
        self.is_leaf = False

        if orientation == "vertical":
            w = self.width // 2
            self.children = [
                KdTreeNode(self.x, self.y, w, self.height, self.level + 1),
                KdTreeNode(self.x + w, self.y, self.width - w, self.height, self.level + 1),
            ]
        else:
            h = self.height // 2
            self.children = [
                KdTreeNode(self.x, self.y, self.width, h, self.level + 1),
                KdTreeNode(self.x, self.y + h, self.width, self.height - h, self.level + 1),
            ]


class KdTree:
    """Manages recursive kd-tree splitting over the same image metrics as the quadtree."""

    def __init__(self, normal_map=None, edge_map=None, min_size=8, dot_threshold=0.98):
        """
        :param normal_map: float32 numpy array HxWx3, assumed to be normalized normals [-1, 1]. (Optional)
        :param edge_map: uint8 numpy array HxW, structurally pre-calculated mask of geometric edges (Optional)
        :param min_size: minimum width/height for a region before stopping to prevent infinite recursion.
        :param dot_threshold: ratio of normals perfectly matching the mean normal.
        """
        self.normal_map = normal_map
        self.edge_map = edge_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold

        if normal_map is not None:
            h, w = normal_map.shape[:2]
        elif edge_map is not None:
            h, w = edge_map.shape[:2]
        else:
            raise ValueError("KdTree requires at least one data matrix cleanly passed.")

        self.root = KdTreeNode(0, 0, w, h)
        self._build(self.root)

    def _build(self, node):
        if node.width <= self.min_size or node.height <= self.min_size:
            return

        if self._should_split(node):
            node.split(self._choose_orientation(node))
            for child in node.children:
                self._build(child)

    def _should_split(self, node):
        x, y, w, h = node.x, node.y, node.width, node.height

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

    def _choose_orientation(self, node):
        """Split the longer side first to keep the rectangles reasonably balanced."""
        if node.width >= node.height:
            return "vertical"
        return "horizontal"

    def draw(self, image, color=(0, 255, 0), thickness=1):
        """Draw the final leaf boxes of the kd-tree overlaid over a given image."""
        img_out = image.copy()

        def _draw_node(node):
            if node.is_leaf:
                cv2.rectangle(img_out, (node.x, node.y), (node.x + node.width, node.y + node.height), color, thickness)
            else:
                for child in node.children:
                    _draw_node(child)

        _draw_node(self.root)
        return img_out


def test_kdtree_workflow():
    """Workflow executing logic purely on provided Ground Truth Normal Map."""
    normal_path = "data/visual_normal.png"
    depth_path = "data/visual_depth.png"

    color_path = "data/img.jpg"
    if not os.path.exists(color_path):
        color_path = "data/frame.0000.color.jpg"

    if not os.path.exists(normal_path) or not os.path.exists(depth_path) or not os.path.exists(color_path):
        print("Skipping KdTree Test visualization, missing required ground truth data logs inside your data folder.")
        return

    print("Executing KdTree pipeline testing specific spatial matrix splits mapped geometrically over your physical RGB Image...")

    color_img_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)

    normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    normals_raw_vectors = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_float = depth_raw.astype(np.float32)

    if len(depth_raw.shape) == 2:
        if depth_raw.dtype == np.uint8:
            depth_8u = depth_raw
        else:
            depth_8u = (depth_float / (np.max(depth_float) + 1e-8) * 255.0).astype(np.uint8)
    else:
        depth_8u = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

    depth_edges_mask = cv2.Canny(depth_8u, threshold1=15, threshold2=50)

    color_gray = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2GRAY)
    rgb_edges_mask = cv2.Canny(color_gray, threshold1=30, threshold2=90)

    combined_edges_mask = cv2.bitwise_or(depth_edges_mask, rgb_edges_mask)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/algorithm_depth_edges_mask.png", depth_edges_mask)
    cv2.imwrite("output/algorithm_rgb_edges_mask.png", rgb_edges_mask)

    print("Generating [Normals Only] boundaries...")
    kt_normals = KdTree(normal_map=normals_raw_vectors, edge_map=None, min_size=4, dot_threshold=0.99)
    out_normals = kt_normals.draw(color_img_bgr, color=(0, 0, 255), thickness=1)

    print("Generating [Depth Edges Only] boundaries...")
    kt_depth_edges = KdTree(normal_map=None, edge_map=depth_edges_mask, min_size=4)
    out_depth_edges = kt_depth_edges.draw(color_img_bgr, color=(255, 0, 0), thickness=1)

    print("Generating [Color Edges Only] boundaries...")
    kt_rgb_edges = KdTree(normal_map=None, edge_map=rgb_edges_mask, min_size=4)
    out_rgb_edges = kt_rgb_edges.draw(color_img_bgr, color=(0, 255, 255), thickness=1)

    print("Generating [All Metrics Combined] boundaries...")
    kt_combined = KdTree(normal_map=normals_raw_vectors, edge_map=combined_edges_mask, min_size=4, dot_threshold=0.99)
    out_combined = kt_combined.draw(color_img_bgr, color=(0, 255, 0), thickness=1)

    cv2.imwrite("output/kdtree_rgb_normals_only.png", out_normals)
    cv2.imwrite("output/kdtree_rgb_depth_edges_only.png", out_depth_edges)
    cv2.imwrite("output/kdtree_rgb_color_edges_only.png", out_rgb_edges)
    cv2.imwrite("output/kdtree_rgb_combined_all.png", out_combined)

    print("Successfully mapped and saved your 4 comparative pipeline algorithms natively drawn over standard RGB images inside the 'output/' folder!")


if __name__ == "__main__":
    test_kdtree_workflow()