import meshcat
import meshcat.geometry as g

class VizServer():

    def __init__(self, port_vis=6000) -> None:
        zmq_url = f'tcp://127.0.0.1:{port_vis}'
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url).open()
        self.mc_vis['scene'].delete()

    def view_pcd(self, pts, colors=None):

        if colors is None:
            colors = pts
        self.mc_vis["scene"].delete()
        self.mc_vis.set_object(g.PointCloud(pts.T, color=colors.T/255.))

    def close(self):
        self.mc_vis.close()