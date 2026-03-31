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
    
    def __init__(self, normal_map, depth_map=None, min_size=8, dot_threshold=0.98, eigen_threshold=0.01):
        """
        :param normal_map: float32 numpy array HxWx3, assumed to be normalized normals [-1, 1].
        :param depth_map: float32 numpy array HxW, depth values. (Optional)
        :param min_size: minimum width/height for a quadrant before stopping to prevent infinite recursion.
        :param dot_threshold: ratio of normals perfectly matching the mean normal.
        :param eigen_threshold: ratio of 3rd eigenvalue of depth distribution to check planarity.
        """
        self.normal_map = normal_map
        self.depth_map = depth_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold
        self.eigen_threshold = eigen_threshold
        
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
            
        N_pts = patch_normals.shape[0] * patch_normals.shape[1]
        
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
            
        # --- 2. Depth Planar Fitting Check (PCA matrix) ---
        if self.depth_map is not None:
            patch_depth = self.depth_map[y:y+h, x:x+w]
            patch_x = self.X[y:y+h, x:x+w].flatten()
            patch_y = self.Y[y:y+h, x:x+w].flatten()
            patch_z = patch_depth.flatten()
            
            # Center points mapping exactly back to geometrical space
            X_c = patch_x - np.mean(patch_x)
            Y_c = patch_y - np.mean(patch_y)
            Z_c = patch_z - np.mean(patch_z)
            
            pts = np.vstack((X_c, Y_c, Z_c)).T
            
            # Calculate array Covariance quickly natively
            cov = np.dot(pts.T, pts) / N_pts
            
            try:
                # Extract eigenvalues
                eigenvalues = np.linalg.eigvalsh(cov)
                
                # Sorted strictly in mathematical ascending order. Lambda 3 identifies the noise structure size.
                lambda_3 = eigenvalues[0]
                total_variance = np.sum(eigenvalues)
                
                if total_variance > 0:
                    planarity_error_ratio = lambda_3 / total_variance
                else:
                    planarity_error_ratio = 0.0
                    
                if planarity_error_ratio > self.eigen_threshold:
                    return True # Points span geometry that is too spherical/bumpy to be a flat plane.
            except np.linalg.LinAlgError:
                return True # Math structure corrupted, branch.
            
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
    
    # Load the specific normal map output generated previously by hdf5 converter
    normal_path = "data/visual_normal.png"
    
    if not os.path.exists(normal_path):
        print(f"Skipping Quadtree Test visualization, {normal_path} not available.")
        return
        
    print("Executing Quadtree pipeline over generated Ground Truth Visual Normal Map...")
    
    # OpenCV implicitly reads standard 0-255 image matrices as BGR
    normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    
    # Decouple the standard 8-bit image back mathematically into numerical float vectors [-1.0, 1.0]
    # Standard geometric translation formula: N = (RGB_Values / 255.0) * 2.0 - 1.0
    normals_raw_vectors = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    print("Evaluating divergence boundaries on Ground Truth data without Depth logic...")
    # Because ground truth rendering sets can be perfectly clean, we can afford a very high dot scalar 0.99
    # Supplying `depth_map=None` explicitly skips the physical structural PCA step
    qt_gt = Quadtree(normals_raw_vectors, depth_map=None, min_size=4, dot_threshold=0.99)
    
    # Draw cleanly in pink specifically representing ground truth boundaries
    overlay_gt = qt_gt.draw(rgb, color=(255, 0, 255), thickness=1)
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/quadtree_gt_viz.png"
    
    # Must explicitly convert back back to BGR for printing!
    overlay_gt_bgr = cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_gt_bgr)

    print(f"Successfully successfully executed Quadtree bounds to {out_path}!")

if __name__ == "__main__":
    test_quadtree_workflow()
