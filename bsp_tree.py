import cv2
import numpy as np
import math
import os

def intersection(p1, p2, l1, l2):
    """Find the intersection of line segment p1->p2 and infinite line l1->l2."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = l1
    x4, y4 = l2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: 
        return None # parallel
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(round(px)), int(round(py)))

def is_left(p, l1, l2):
    """Determine if point p is strictly on the 'left' side of line l1->l2."""
    return (l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1] - l1[1]) * (p[0] - l1[0]) >= 0

def split_polygon(poly, l1, l2):
    """Clips a polygon against an infinite line forming two convex child polygons."""
    poly_left = []
    poly_right = []
    
    for i in range(len(poly)):
        curr_p = poly[i]
        prev_p = poly[i - 1]
        
        curr_left = is_left(curr_p, l1, l2)
        prev_left = is_left(prev_p, l1, l2)
        
        if curr_left != prev_left:
            ix = intersection(prev_p, curr_p, l1, l2)
            if ix is not None:
                poly_left.append(ix)
                poly_right.append(ix)
                
        if curr_left:
            poly_left.append(curr_p)
        else:
            poly_right.append(curr_p)
            
    # Simple deduplication just in case an exact point hit generated duplicated vertices
    left_dedup = []
    for p in poly_left:
        if not left_dedup or p != left_dedup[-1]: left_dedup.append(p)
    if len(left_dedup) > 0 and left_dedup[0] == left_dedup[-1]:
       left_dedup.pop()
       
    right_dedup = []
    for p in poly_right:
        if not right_dedup or p != right_dedup[-1]: right_dedup.append(p)
    if len(right_dedup) > 0 and right_dedup[0] == right_dedup[-1]:
       right_dedup.pop()

    return left_dedup, right_dedup

class BSPTreeNode:
    def __init__(self, polygon, level=0):
        self.polygon = polygon 
        self.level = level
        self.children = []
        self.is_leaf = True

class BSPTree:
    def __init__(self, normal_map=None, edge_map=None, min_area=32, dot_threshold=0.98, max_depth=15):
        self.normal_map = normal_map
        self.edge_map = edge_map
        self.min_area = min_area
        self.dot_threshold = dot_threshold
        self.max_depth = max_depth
        
        if normal_map is not None:
            self.h, self.w = normal_map.shape[:2]
        elif edge_map is not None:
            self.h, self.w = edge_map.shape[:2]
        else:
            raise ValueError("BSPTree requires at least one data matrix cleanly passed.")
            
        root_polygon = [(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)]
        self.root = BSPTreeNode(root_polygon)
        self._build(self.root)
        
    def _build(self, node):
        if node.level >= self.max_depth:
            return
            
        pts = np.array(node.polygon, dtype=np.int32)
        if len(pts) < 3:
            return # Impossible invalid geometry
            
        # Calculate Bounding Box of Polygon for localized logic
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.w, x_max), min(self.h, y_max)
        
        patch_w = x_max - x_min
        patch_h = y_max - y_min
        
        # Fallback termination
        if patch_w * patch_h <= self.min_area:
            return
            
        # Create tight localized mask
        local_pts = pts - [x_min, y_min]
        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts], 255)
        
        should_split = self._should_split(x_min, y_min, patch_w, patch_h)
                
        if should_split:
            poly1, poly2 = self._split_shape(node.polygon, mask, x_min, y_min, patch_w, patch_h)
            
            if poly1 is not None and poly2 is not None:
                node.is_leaf = False
                child1 = BSPTreeNode(poly1, node.level + 1)
                child2 = BSPTreeNode(poly2, node.level + 1)
                node.children = [child1, child2]
                self._build(child1)
                self._build(child2)

    def _should_split(self, x, y, w, h):
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
                
    def _split_shape(self, polygon, mask, x_min, y_min, patch_w, patch_h):
        """Finds geometric splitting hyperplanes using Edge Detection/Hough Lines inside Polygon bounds."""
        lines = None
        has_local_edges = False
        
        if self.edge_map is not None:
            patch_edges = self.edge_map[y_min:y_min+patch_h, x_min:x_min+patch_w]
            masked_edges = cv2.bitwise_and(patch_edges, patch_edges, mask=mask)
            
            # Identify prominent physical edges purely inside this shape
            if np.any(masked_edges > 0):
                 # Tuning parameters to find structural lines vs noise
                 min_lin_len = max(5, int(min(patch_w, patch_h) * 0.2))
                 lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=min_lin_len, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # Pick strongest line
            best_line = None
            max_len = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = (x2-x1)**2 + (y2-y1)**2
                if length > max_len:
                    max_len = length
                    best_line = (x1, y1, x2, y2)
                    
            if best_line:
                lx1, ly1, lx2, ly2 = best_line
                l1 = (lx1 + x_min, ly1 + y_min)
                l2 = (lx2 + x_min, ly2 + y_min)
                
                # Check line isn't a single point (geometry bug)
                if l1 != l2:
                    p1, p2 = split_polygon(polygon, l1, l2)
                    
                    # Ensure the split actually divided the polygon meaningfully
                    # By checking that both children have at least 5% of the bounding area
                    parent_area = patch_w * patch_h
                    if len(p1) > 2 and len(p2) > 2:
                        area1 = cv2.contourArea(np.array(p1, dtype=np.float32))
                        area2 = cv2.contourArea(np.array(p2, dtype=np.float32))
                        
                        if area1 > 0.05 * parent_area and area2 > 0.05 * parent_area:
                            return p1, p2

        # FALLBACK: If explicit Hough Lines failed or we didn't use an edge map but variance caused a split.
        # Spilt it geometrically by centroid (equivalent to quadtree but only yielding 2 halves)
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        
        if patch_w > patch_h: # Split Vertically
            l1 = (cx, 0)
            l2 = (cx, self.h)
        else: # Split Horizontally
            l1 = (0, cy)
            l2 = (self.w, cy)
            
        p1, p2 = split_polygon(polygon, l1, l2)
        if len(p1) > 2 and len(p2) > 2:
            return p1, p2
            
        return None, None
        
    def draw(self, image, color=(0, 255, 0), thickness=1):
        """Draws the final leaf polygons of the BSP tree."""
        img_out = image.copy()
        
        def _draw_node(node):
            if node.is_leaf:
                pts = np.array(node.polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_out, [pts], isClosed=True, color=color, thickness=thickness)
            else:
                for child in node.children:
                    _draw_node(child)
                    
        _draw_node(self.root)
        return img_out

def test_bsp_workflow():
    normal_path = "data/visual_normal.png"
    depth_path = "data/visual_depth.png"
    
    color_path = "data/img.jpg"
    if not os.path.exists(color_path):
        color_path = "data/frame.0000.color.jpg"
    
    if not os.path.exists(normal_path) or not os.path.exists(depth_path) or not os.path.exists(color_path):
        print("Missing required ground truth data logs inside your data folder.")
        return
        
    print("Executing BSP Tree geometric pipeline mapping...")
    color_img_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)

    # Load Normals
    normal_bgr = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    normals_raw_vectors = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # Process Depth
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
    
    print("Generating [Normals Only] polygons...")
    bsp_normals = BSPTree(normal_map=normals_raw_vectors, edge_map=None, min_area=32, dot_threshold=0.99)
    out_normals = bsp_normals.draw(color_img_bgr, color=(0, 0, 255), thickness=1)
    
    print("Generating [Depth Edges Only] polygons...")
    bsp_depth = BSPTree(normal_map=None, edge_map=depth_edges_mask, min_area=32)
    out_depth = bsp_depth.draw(color_img_bgr, color=(255, 0, 0), thickness=1)
    
    print("Generating [Color Edges Only] polygons...")
    bsp_rgb = BSPTree(normal_map=None, edge_map=rgb_edges_mask, min_area=32)
    out_rgb = bsp_rgb.draw(color_img_bgr, color=(0, 255, 255), thickness=1)
    
    print("Generating [All Metrics Combined] polygons...")
    bsp_comb = BSPTree(normal_map=normals_raw_vectors, edge_map=combined_edges_mask, min_area=32, dot_threshold=0.99)
    out_comb = bsp_comb.draw(color_img_bgr, color=(0, 255, 0), thickness=1)
    
    cv2.imwrite("output/bsp_rgb_normals_only.png", out_normals)
    cv2.imwrite("output/bsp_rgb_depth_edges_only.png", out_depth)
    cv2.imwrite("output/bsp_rgb_color_edges_only.png", out_rgb)
    cv2.imwrite("output/bsp_rgb_combined_all.png", out_comb)
    
    print("Saved 4 comparative pipeline algorithms natively mapped via continuous geometric splitting in the 'output/' folder!")

if __name__ == "__main__":
    test_bsp_workflow()
