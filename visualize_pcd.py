import open3d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname")

    args = parser.parse_args()
    fname = args.fname
    pcd = open3d.io.read_point_cloud(fname)
    open3d.visualization.draw_geometries([pcd])