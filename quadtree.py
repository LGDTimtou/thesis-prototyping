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
    
    def __init__(self, normal_map, depth_map=None, min_size=8, dot_threshold=0.98, depth_error_threshold=10.0):
        """
        :param normal_map: float32 numpy array HxWx3, assumed to be normalized normals [-1, 1].
        :param depth_map: float32 numpy array HxW, depth values. (Optional)
        :param min_size: minimum width/height for a quadrant before stopping to prevent infinite recursion.
        :param dot_threshold: ratio of normals perfectly matching the mean normal.
        :param depth_error_threshold: threshold for Mean Squared Error of planar fit on depth values.
        """
        self.normal_map = normal_map
        self.depth_map = depth_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold
        self.depth_error_threshold = depth_error_threshold
        
        # Determine shape accurately regardless of depth_map presence
        h, w = normal_map.shape[:2]
        
        # Precompute coordinate grids for plane fitting math to keep evaluation ultra-fast
        if self.depth_map is not None:
            self.X, self.Y = np.meshgrid(np.arange(w), np.arange(h))
        
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
        patch_normals = self.normal_map[y:y+h, x:x+w]
        
        if patch_normals.size == 0:
            return False
            
        # Calculate the pure average normal
        mean_normal = np.mean(patch_normals, axis=(0, 1))
        norm_len = np.linalg.norm(mean_normal)
        
        if norm_len > 0:
            mean_normal /= norm_len
            
        # Dot product array: element wise multiply and sum over 3rd axis
        dots = np.sum(patch_normals * mean_normal, axis=2)
        mean_dot = np.mean(dots)
        
        if mean_dot < self.dot_threshold:
            return True # Values represent too sharp an angle from the average! Split.
            
        # --- 2. Depth Planar Fitting Check (Least Squares) ---
        # PCA fails on huge depth jumps because it treats all 3 axes the same.
        # A massive cliff causes PCA to just rotate its primary axis straight down Z, 
        # making the "thickness" appear small. 
        # We must explicitly regress Z against X and Y using standard plane fitting!
        if self.depth_map is not None:
            patch_depth = self.depth_map[y:y+h, x:x+w]
            patch_x = self.X[y:y+h, x:x+w].flatten()
            patch_y = self.Y[y:y+h, x:x+w].flatten()
            patch_z = patch_depth.flatten()
            
            # Construct least-squares matrix: Z = aX + bY + c
            A = np.c_[patch_x, patch_y, np.ones(patch_x.shape[0])]
            
            try:
                # Solve for plane coefficients [a, b, c]
                coeffs, residuals, rank, s = np.linalg.lstsq(A, patch_z, rcond=None)
                
                # If there are residuals, we calculate the Mean Squared Error against the flat plane
                if len(residuals) > 0:
                    mse = residuals[0] / len(patch_z)
                else:
                    mse = 0.0
                    
                if mse > self.depth_error_threshold:
                    return True # Geometry contains a cliff or curve too severe to approximate
                    
            except np.linalg.LinAlgError:
                return True # Math structure corrupted out of weird complexity, branch.
            
        # Both variance and optional plane approximation are perfectly acceptable! Save the node intact.
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
    
    if not os.path.exists(normal_path) or not os.path.exists(depth_path):
        print(f"Skipping Quadtree Test visualization, missing required ground truth data logs.")
        return
        
    print("Executing Quadtree pipeline over generated Ground Truth Visual Maps...")
    
    # 1. Load the Normal Map vectors mathematically
    normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    
    # Standard geometric translation formula: N = (RGB_Values / 255.0) * 2.0 - 1.0
    normals_raw_vectors = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # 2. Load the Depth Map natively without capping formats
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_float = depth_raw.astype(np.float32)
    
    # In PCA, variances of drastically different scales break the math matrices!
    # (e.g., Matrix X/Y spans ~1000px, but depth spans [0.0 - 1.0] or [0 - 65535]).
    # We cleanly normalize depth values specifically back to physical image coordinate boundaries.
    img_max_dim = max(rgb.shape[0], rgb.shape[1])
    depth_pca_mapped = (depth_float / (np.max(depth_float) + 1e-8)) * img_max_dim

    # Produce a strict 8-bit visual rendering map to draw perfectly visible colored boxes on
    if len(depth_raw.shape) == 2:
        if depth_raw.dtype == np.uint8:
            depth_8u = depth_raw
        else:
            # Safely cast down huge 16-bit variants back to an 8-bit visible surface
            depth_8u = (depth_float / (np.max(depth_float) + 1e-8) * 255.0).astype(np.uint8)
        depth_vis_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
    else:
        # Assuming it is already safely 3 channel
        depth_vis_bgr = depth_raw.copy()
    
    # --- 3. Run Logic WITHOUT depth (Normals Only) ---
    print("Evaluating divergence boundaries WITHOUT depth logic...")
    qt_no_depth = Quadtree(normals_raw_vectors, depth_map=None, min_size=4, dot_threshold=0.99)
    # Using RED to indicate it had no underlying depth structural bounds
    normal_img_nodepth_overlay = qt_no_depth.draw(normal_bgr, color=(0, 0, 255), thickness=1)
    depth_img_nodepth_overlay  = qt_no_depth.draw(depth_vis_bgr, color=(0, 0, 255), thickness=1)

    # --- 4. Run Logic WITH depth logic ---
    print("Evaluating divergence boundaries WITH underlying mapped depth logic...")
    # Add a reasonably strict standard plane threshold ratio since our depth map natively spans the image bounds
    qt_with_depth = Quadtree(normals_raw_vectors, depth_map=depth_pca_mapped, min_size=4, dot_threshold=0.99, depth_error_threshold=1.0)
    # GREEN successfully represents depth+variance approved splits
    normal_img_withdepth_overlay = qt_with_depth.draw(normal_bgr, color=(0, 255, 0), thickness=1)
    depth_img_withdepth_overlay  = qt_with_depth.draw(depth_vis_bgr, color=(0, 255, 0), thickness=1)
    
    # --- 5. Export Overlays ---
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/quadtree_nodepth_normals_viz.png", normal_img_nodepth_overlay)
    cv2.imwrite("output/quadtree_nodepth_depth_viz.png", depth_img_nodepth_overlay)
    cv2.imwrite("output/quadtree_withdepth_normals_viz.png", normal_img_withdepth_overlay)
    cv2.imwrite("output/quadtree_withdepth_depth_viz.png", depth_img_withdepth_overlay)

    print("Successfully explicitly executed all comparison Quadtree bounds to the 'output/' folder!")

if __name__ == "__main__":
    test_quadtree_workflow()
