from plyfile import PlyData
import numpy as np


def print_first_xyz_points(input_ply_path, max_points=100):
    plydata = PlyData.read(input_ply_path)
    vertex_data = plydata["vertex"]

    print("Available vertex properties:")
    print(vertex_data.data.dtype.names)

    xyz = np.stack(
        (vertex_data["x"], vertex_data["y"], vertex_data["z"]),
        axis=-1,
    )

    count = min(max_points, xyz.shape[0])
    print(f"\nTotal points: {xyz.shape[0]}")
    print(f"First {count} XYZ points:")

    for i in range(count):
        x, y, z = xyz[i]
        print(f"{i:03d}: x={x:.6f}, y={y:.6f}, z={z:.6f}")


if __name__ == "__main__":
    input_ply_path = r"C:\Users\lisal\Dropbox\TIMON\Informatica2Ma\Thesis\project\resultaten\20260302_115012_dc-on_sam-tile_opt-on_maxnew-inf\point_cloud.ply"
    print_first_xyz_points(input_ply_path=input_ply_path, max_points=100)
