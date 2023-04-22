import meshcat
import meshcat.geometry as g

class VizServer():

    def __init__(self, port_vis=6000) -> None:
        zmq_url = f'tcp://127.0.0.1:{port_vis}'
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url).open()
        self.mc_vis['scene'].delete()
        self.mc_vis['/'].delete()

    def view_pcd(self, pts, colors=None, name="scene"):

        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(g.PointCloud(pts.T, color=colors.T/255.))

    def close(self):

        self.mc_vis.close()


import open3d
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname")

    args = parser.parse_args()
    fname = args.fname
    pcd = open3d.io.read_point_cloud(fname)

    pts = np.asarray(pcd.points)

    vis = VizServer()
    vis.view_pcd(pts)


    # open3d.visualization.draw_geometries([pcd])