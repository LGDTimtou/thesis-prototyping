import numpy as np


class TreeNode:
    def __init__(self, level=0):
        self.level = level
        self.children = []
        self.is_leaf = True


class Tree:
    def __init__(self, normal_map=None, edge_map=None, min_size=8, dot_threshold=0.98):
        self.normal_map = normal_map
        self.edge_map = edge_map
        self.min_size = min_size
        self.dot_threshold = dot_threshold

        if normal_map is not None:
            self.h, self.w = normal_map.shape[:2]
        elif edge_map is not None:
            self.h, self.w = edge_map.shape[:2]
        else:
            raise ValueError(f"{self.__class__.__name__} requires at least one data matrix cleanly passed.")

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