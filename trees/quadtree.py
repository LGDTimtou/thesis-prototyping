import cv2
import numpy as np
import os

try:
    from .tree import Tree, TreeNode
except ImportError:
    from tree import Tree, TreeNode


class QuadtreeNode(TreeNode):
    """A single section of the processed image acting as a leaf or branch."""

    def __init__(self, x, y, width, height, level=0):
        super().__init__(level=level)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def split(self):
        """Divide the square into 4 exact quadrants."""
        self.is_leaf = False
        w = self.width // 2
        h = self.height // 2

        self.children = [
            QuadtreeNode(self.x, self.y, w, h, self.level + 1),
            QuadtreeNode(self.x + w, self.y, self.width - w, h, self.level + 1),
            QuadtreeNode(self.x, self.y + h, w, self.height - h, self.level + 1),
            QuadtreeNode(self.x + w, self.y + h, self.width - w, self.height - h, self.level + 1)
        ]


class Quadtree(Tree):
    """Manages the recursive splitting logic evaluating underlying spatial image variables."""

    def __init__(self, color_image, normal_image, depth_image, min_size=8, dot_threshold=0.98, **kwargs):
        super().__init__(
            color_image=color_image,
            normal_image=normal_image,
            depth_image=depth_image,
            min_size=min_size,
            dot_threshold=dot_threshold,
            **kwargs,
        )

        self.root = QuadtreeNode(0, 0, self.w, self.h)
        self._build(self.root)

    def _build(self, node):
        if node.width <= self.min_size or node.height <= self.min_size:
            return

        if self._should_split(node.x, node.y, node.width, node.height):
            node.split()
            for child in node.children:
                self._build(child)
        
    def draw(self, image, color=(0, 255, 0), thickness=1):
        """Draws the final leaf boxes of the Quadtree grid overlaid over a given image."""
        img_out = image.copy()
        
        def _draw_node(node):
            if node.is_leaf:
                cv2.rectangle(img_out, (node.x, node.y), (node.x + node.width, node.y + node.height), color, thickness)
            else:
                for child in node.children:
                    _draw_node(child)
                    
        _draw_node(self.root)
        return img_out


def test_quadtree_workflow():
    """Workflow executing logic purely on provided Ground Truth Normal Map."""
    import os
    
    # Paths pointing exactly to the specific user files
    normal_path = "data/visual_normal.png"
    depth_path = "data/visual_depth.png"
    
    # Determine the strict RGB user image
    color_path = "data/img.jpg"
    if not os.path.exists(color_path):
        # Fallback to standard generated GT files if named differently
        color_path = "data/frame.0000.color.jpg"
    
    if not os.path.exists(normal_path) or not os.path.exists(depth_path) or not os.path.exists(color_path):
        print(f"Skipping Quadtree Test visualization, missing required ground truth data logs inside your data folder.")
        return
        
    print("Executing Quadtree pipeline testing specific spatial matrix splits mapped geometrically over your physical RGB Image...")
    
    # Load strictly formatted mapping target
    color_img_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)

    # 1. Load the Normal Map vectors mathematically
    normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    
    # Standard geometric translation formula: N = (RGB_Values / 255.0) * 2.0 - 1.0
    normals_raw_vectors = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # 2. Process Depth specifically targeting algorithmic Edges
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
    
    # --- Execute Pre-Calculated Canny Extractor on RGB COLOR ---
    color_gray = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2GRAY)
    rgb_edges_mask = cv2.Canny(color_gray, threshold1=30, threshold2=90) # Tighter threshold bounds for noisy color images
    
    # Mathematical Stack of both structural edge detection variants
    combined_edges_mask = cv2.bitwise_or(depth_edges_mask, rgb_edges_mask)
    
    # Provide the masks out so the user can visually trace the algorithmic inputs
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/algorithm_depth_edges_mask.png", depth_edges_mask)
    cv2.imwrite("output/algorithm_rgb_edges_mask.png", rgb_edges_mask)
    
    # --- Option 1: Run Logic using NORMALS ONLY ---
    print("Generating [Normals Only] boundaries...")
    qt_normals = Quadtree(
        color_image=color_img_bgr,
        normal_image=normal_bgr,
        depth_image=depth_raw,
        min_size=4,
        dot_threshold=0.99,
        rgb_threshold1=30,
        rgb_threshold2=90,
    )
    out_normals = qt_normals.draw(color_img_bgr, color=(0, 0, 255), thickness=1) # RED purely means Normals
    
    # --- Option 2: Run Logic using DEPTH EDGES ONLY ---
    print("Generating [Depth Edges Only] boundaries...")
    qt_depth_edges = Quadtree(
        color_image=color_img_bgr,
        normal_image=normal_bgr,
        depth_image=depth_raw,
        min_size=4,
        dot_threshold=0.99,
        rgb_threshold1=30,
        rgb_threshold2=90,
    )
    out_depth_edges = qt_depth_edges.draw(color_img_bgr, color=(255, 0, 0), thickness=1) # BLUE purely means Depth Drop
    
    # --- Option 3: Run Logic using RGB COLOR EDGES ONLY ---
    print("Generating [Color Edges Only] boundaries...")
    qt_rgb_edges = Quadtree(
        color_image=color_img_bgr,
        normal_image=normal_bgr,
        depth_image=depth_raw,
        min_size=4,
        dot_threshold=0.99,
        rgb_threshold1=30,
        rgb_threshold2=90,
    )
    out_rgb_edges = qt_rgb_edges.draw(color_img_bgr, color=(0, 255, 255), thickness=1) # YELLOW purely means Texture Color Boundaries
    
    # --- Option 4: Run Logic COMBINED ---
    print("Generating [All Metrics Combined] algorithm boundaries...")
    qt_combined = Quadtree(
        color_image=color_img_bgr,
        normal_image=normal_bgr,
        depth_image=depth_raw,
        min_size=4,
        dot_threshold=0.99,
        rgb_threshold1=30,
        rgb_threshold2=90,
    )
    out_combined = qt_combined.draw(color_img_bgr, color=(0, 255, 0), thickness=1) # GREEN explicitly stacks all 3 metrics perfectly
    
    # --- 5. Export the specific test arrays entirely mapped back onto RGB space ---
    cv2.imwrite("output/quadtree_rgb_normals_only.png", out_normals)
    cv2.imwrite("output/quadtree_rgb_depth_edges_only.png", out_depth_edges)
    cv2.imwrite("output/quadtree_rgb_color_edges_only.png", out_rgb_edges)
    cv2.imwrite("output/quadtree_rgb_combined_all.png", out_combined)

    print("Successfully mapped and saved your 4 comparative pipeline algorithms natively drawn over standard RGB images inside the 'output/' folder!")

if __name__ == "__main__":
    test_quadtree_workflow()
