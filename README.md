# Thesis Prototype Commands (Short Guide)

This project contains scripts for:
- VP-based image upright rotation
- KD-tree + single splat generation
- Full multi-tree parameter sweeps

## 1) Rotate input image upright (VP-based)

Run:

```bash
python -m scripts.vp_rotate_upright
```

Writes:
- `output/img_upright_vp.jpg`
- `output/img_upright_vp_metadata.json`

The metadata file stores the upright rotation angle and is used later for rotating results back.

## 2) Run KD-tree + single splats on upright image (then rotate back)

Run:

```bash
python -m scripts.run_kdtree_single_splat_upright
```

Behavior:
- If upright image or metadata is missing, it automatically runs the VP rotation step first.
- Runs KD-tree splitting and single splat extraction.
- Rotates the splat outputs back to the original orientation.

Main outputs:
- `output/upright_kdtree_edge_map.png`
- `output/upright_kdtree_rgb_color_edges_only.png`
- `output/upright_kdtree_splats_clean.png`
- `output/upright_kdtree_splats_with_boxes.png`
- `output/upright_kdtree_splats_clean_backrotated.png`
- `output/upright_kdtree_splats_with_boxes_backrotated.png`
- `output/upright_kdtree_splats_count.json`

## 3) Run full single-splat sweep with tree selection

Script:
- `splats/single_splat.py`

Default (runs all trees and all parameter settings):

```bash
python -m splats.single_splat
```

Use `--trees` to run specific tree(s):

```bash
python -m splats.single_splat --trees kdtree
python -m splats.single_splat --trees kdtree bsptree
python -m splats.single_splat --trees kdtree,bsptree
```

Valid `--trees` values:
- `quadtree`
- `kdtree`
- `bsptree`

## Notes

- The sweep stores results in `results/` using folders like:
  - `results/kdtree_min_size-4_threshold_30-90`
  - `results/bsptree_min_size-4_threshold_30-90`
- If your environment uses a specific Python interpreter, run commands from that environment.
