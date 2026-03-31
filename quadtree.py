import cv2
import numpy as np
import math

class QuadtreeNode:
    """A single section of the processed image acting as a leaf or branch."""
    def __init__(self, x, y, width, height, level=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.level = level
        self.children = []
        self.is_leaf = True
        
    def split(self):
        """Divide the square into 4 exact quadrants."""
        self.is_leaf = False
        w = self.width // 2
        h = self.height // 2
        
        # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        self.children = [
            QuadtreeNode(self.x, self.y, w, h, self.level + 1),
            QuadtreeNode(self.x + w, self.y, self.width - w, h, self.level + 1),
            QuadtreeNode(self.x, self.y + h, w, self.height - h, self.level + 1),
            QuadtreeNode(self.x + w, self.y + h, self.width - w, self.height - h, self.level + 1)
        ]

class Quadtree:
    """Manages the recursive splitting logic evaluating underlying spatial image variables."""
    
    def __init__(self, normal_map=None, edge_map=None, min_size=8, dot_threshold=0.98):
        """
        :param normal_map: float32 numpy array HxWx3, assumed to be normalized normals [-1, 1]. (Optional)
        :param edge_map: uint8 numpy array HxW, structurally pre-calculated mask of geometric edges (Optional)
        :param min_size: minimum width/height for a quadrant before stopping to prevent infinite recursion.
        :param dot_threshold: ratio of normals perfectly matching the mean normal.
        """
        self.normal_map = normal_map
        self.edge_map = edge_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold
        
        # Determine shape accurately regardless of whether we drop normal maps or edge maps entirely
        if normal_map is not None:
            h, w = normal_map.shape[:2]
        elif edge_map is not None:
            h, w = edge_map.shape[:2]
        else:
            raise ValueError("Quadtree requires at least one data matrix cleanly passed.")
            
        # Root encompasses the entire matrix
        self.root = QuadtreeNode(0, 0, w, h)
        self._build(self.root)
        
    def _build(self, node):
        # Base case: We reached our resolution limit
        if node.width <= self.min_size or node.height <= self.min_size:
            return
            
        if self._should_split(node):
            node.split()
            for child in node.children:
                self._build(child)
                
    def _should_split(self, node):
        x, y, w, h = node.x, node.y, node.width, node.height
        
        # --- 1. Normal Variance Check (Dot Product metric) ---
        if self.normal_map is not None:
            patch_normals = self.normal_map[y:y+h, x:x+w]
            
            if patch_normals.size > 0:
                # Calculate the pure average normal
                mean_normal = np.mean(patch_normals, axis=(0, 1))
                norm_len = np.linalg.norm(mean_normal)
                
                if norm_len > 0:
                    mean_normal /= norm_len
                    
                # Dot product array: element wise multiply and sum over 3rd axis
                dots = np.sum(patch_normals * mean_normal, axis=2)
                if np.mean(dots) < self.dot_threshold:
                    return True # Values represent too sharp an angle from the average! Split.
            
        # --- 2. Structural Edge Map Check ---
        # Instead of heavy Least Squares matrices, we execute an O(1) instantaneous lookup
        # across the globally computed edge pixels.
        if self.edge_map is not None:
            patch_edges = self.edge_map[y:y+h, x:x+w]
            
            # If the patch contains any strict physical cliff boundary (a white edge pixel), drop the node!
            if np.any(patch_edges > 0):
                return True 
            
        # Both variance and optional edge approximation are perfectly acceptable! Save the node intact.
        return False
        
    def draw(self, image, color=(0, 255, 0), thickness=1):
        """Draws the final leaf boxes of the Quadtree grid overlaid over a given image."""
        img_out = image.copy()
        
        def _draw_node(node):
            if node.is_leaf:
                # Draw the square
                cv2.rectangle(img_out, (node.x, node.y), (node.x + node.width, node.y + node.height), color, thickness)
            else:
                # Move to children branches
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
    rgb_edges_mask = cv2.Canny(color_gray, threshold1=50, threshold2=150) # Tighter threshold bounds for noisy color images
    
    # Mathematical Stack of both structural edge detection variants
    combined_edges_mask = cv2.bitwise_or(depth_edges_mask, rgb_edges_mask)
    
    # Provide the masks out so the user can visually trace the algorithmic inputs
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/algorithm_depth_edges_mask.png", depth_edges_mask)
    cv2.imwrite("output/algorithm_rgb_edges_mask.png", rgb_edges_mask)
    
    # --- Option 1: Run Logic using NORMALS ONLY ---
    print("Generating [Normals Only] boundaries...")
    qt_normals = Quadtree(normal_map=normals_raw_vectors, edge_map=None, min_size=4, dot_threshold=0.99)
    out_normals = qt_normals.draw(color_img_bgr, color=(0, 0, 255), thickness=1) # RED purely means Normals
    
    # --- Option 2: Run Logic using DEPTH EDGES ONLY ---
    print("Generating [Depth Edges Only] boundaries...")
    qt_depth_edges = Quadtree(normal_map=None, edge_map=depth_edges_mask, min_size=4)
    out_depth_edges = qt_depth_edges.draw(color_img_bgr, color=(255, 0, 0), thickness=1) # BLUE purely means Depth Drop
    
    # --- Option 3: Run Logic using RGB COLOR EDGES ONLY ---
    print("Generating [Color Edges Only] boundaries...")
    qt_rgb_edges = Quadtree(normal_map=None, edge_map=rgb_edges_mask, min_size=4)
    out_rgb_edges = qt_rgb_edges.draw(color_img_bgr, color=(0, 255, 255), thickness=1) # YELLOW purely means Texture Color Boundaries
    
    # --- Option 4: Run Logic COMBINED ---
    print("Generating [All Metrics Combined] algorithm boundaries...")
    qt_combined = Quadtree(normal_map=normals_raw_vectors, edge_map=combined_edges_mask, min_size=4, dot_threshold=0.99)
    out_combined = qt_combined.draw(color_img_bgr, color=(0, 255, 0), thickness=1) # GREEN explicitly stacks all 3 metrics perfectly
    
    # --- 5. Export the specific test arrays entirely mapped back onto RGB space ---
    cv2.imwrite("output/quadtree_rgb_normals_only.png", out_normals)
    cv2.imwrite("output/quadtree_rgb_depth_edges_only.png", out_depth_edges)
    cv2.imwrite("output/quadtree_rgb_color_edges_only.png", out_rgb_edges)
    cv2.imwrite("output/quadtree_rgb_combined_all.png", out_combined)

    print("Successfully mapped and saved your 4 comparative pipeline algorithms natively drawn over standard RGB images inside the 'output/' folder!")

if __name__ == "__main__":
    test_quadtree_workflow()
