import meshcat
import meshcat.geometry as g

# from pyngrok import ngrok
# http_tunnel = ngrok.connect(meshcat.port(), bind_tls=False)
# web_url = http_tunnel.public_url

# set_log_level(prev_log_level)
# print(f'Meshcat is now available at {web_url}')


class VizServer():

    def __init__(self, port_vis=6000) -> None:
        zmq_url = f'tcp://127.0.0.1:{port_vis}'
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url).open()
        self.mc_vis['scene'].delete()
        self.mc_vis['meshcat'].delete()
        self.mc_vis['/'].delete()

    def view_pcd(self, pts, colors=None, name="scene"):

        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(g.PointCloud(pts.T, color=colors.T/255.))

    def view_grasps(self, poses, name=None, freq=100):
        """
        shows a sample grasp as represented by a little fork guy
        freq: show one grasp per every (freq) grasps (1/freq is ratio of visualized grasps)
        """

        control_pt_list = self.get_key_points(poses)
        if name is None:
            name = 'scene/grasp_poses/'

        for i, gripper in enumerate(control_pt_list):
            color = np.zeros_like(gripper) * 255
            
            gripper = gripper[1:,:]
            gripper = gripper[[2, 0, 1, 3], :]
            gripper = np.transpose(gripper, axes=(1,0))
            
            name_i = 'pose'+str(i)
            if i%freq == 0:
                self.mc_vis["scene/grasp_poses/" + name+ "/" + name_i].set_object(g.Line(g.PointsGeometry(gripper)))

    def get_key_points(self, poses, include_sym=False):
        import grasping.model.mesh_utils as mesh_utils

        gripper_object = mesh_utils.create_gripper('panda', root_folder='./grasping')
        gripper_np = gripper_object.get_control_point_tensor(poses.shape[0])
        hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
        gripper_pts = np.concatenate((gripper_np, hom), 2).transpose(0,2,1)
        pts = np.matmul(poses, gripper_pts).transpose(0,2,1)
        return pts

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