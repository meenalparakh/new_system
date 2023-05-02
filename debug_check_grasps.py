from visualize_pcd import VizServer
from robot_env import MyRobot, print_object_dicts
import numpy as np
from grasping.eval import cgn_infer

if __name__ == "__main__":
    robot = MyRobot(gui=True, grasper=True, clip=True)
    robot.reset("cup_over_bowl")
    # robot.reset("one_object")

    obs = robot.get_obs()

    combined_pts, combined_rgb = robot.get_combined_pcd(
        obs["colors"], obs["depths"], idx=None
    )
    combined_pts, combined_rgb, _ = robot.crop_pcd(combined_pts, combined_rgb, None)

    # floor_mask = (combined_pts[:, 2] < 1.0)
    # floor = combined_pts[floor_mask]

    # ////////////////////////////////////////////////////////////////////////////////////////
    # mask = (combined_pts[:, 2] > 0.95).reshape((-1, 1))

    viz = VizServer()
    viz.view_pcd(combined_pts, combined_rgb)

    # ht = np.min(combined_pts[:, 2])
    # x, y = np.mean(combined_pts[:, :2], axis=0)
    # translation = np.array([x, y, ht])
    # pcd = combined_pts - translation
    # pred_grasps, pred_success, _ = cgn_infer(
    #     robot.grasper, pcd, mask, threshold=0.85
    # )
    # pred_grasps[:, :3, 3] = pred_grasps[:, :3, 3] + translation
    # print(len(pred_grasps), pred_grasps.shape, "<= these are the grasps")

    # n = 0 if pred_success is None else pred_grasps.shape[0]
    # print('model pass.', n, 'grasps found.')

    # viz.view_grasps(pred_grasps, "check", freq=1)
    # ////////////////////////////////////////////////////////////////////////////////////////

    segs, info_dict = robot.get_segment_labels_and_embeddings(
        obs["colors"], obs["depths"], robot.clip
    )

    object_dicts = robot.get_segmented_pcd(
        obs["colors"],
        obs["depths"],
        segs,
        remove_floor_ht=1.0,
        std_threshold=0.02,
        label_infos=info_dict,
        visualization=True,
        process_pcd_fn=robot.crop_pcd,
    )

    description, object_dicts = robot.get_scene_description(object_dicts)

    robot.object_dicts = object_dicts
    print_object_dicts(object_dicts)
    print(description)

    # # //////////////////////////////////////////////////////////////////////////////
    # # Visualization of point clouds
    # # //////////////////////////////////////////////////////////////////////////////

    print("Number of objects", len(object_dicts))
    pcds = []
    colors = []
    # rand_colors = dp.get_colors(len(object_dicts))
    for idx, obj in enumerate(object_dicts):
        label = object_dicts[obj]["label"][0]
        pcd = object_dicts[obj]["pcd"]
        color = object_dicts[obj]["rgb"]

        viz.view_pcd(pcd, color, f"{idx}_{label}")
        pcds.append(pcd)
        colors.append(color)

    pcds = np.vstack(pcds)
    colors = np.vstack(colors)
    viz.view_pcd(pcds, colors)

    # # //////////////////////////////////////////////////////////////////////////////
    # # Grasps
    # # //////////////////////////////////////////////////////////////////////////////

    # all_grasps = []
    # all_scores = []
    # for obj_id in robot.object_dicts:
    #     grasps, scores = robot.get_grasp(
    #         obj_id, threshold=0.95, add_floor=robot.bg_pcd
    #     )

    #     if scores is None:
    #         print("No grasps to show.")

    #     else:
    #         all_grasps.append(grasps)
    #         all_scores.append(scores)
    #         best_id = np.argmax(scores)
    #         chosen_grasp = grasps[best_id : best_id + 1]
    #         chosen_grasp = grasps
    #         viz.view_grasps(
    #             chosen_grasp,
    #             name=robot.object_dicts[obj_id]["used_name"].replace(" ", "_"),
    #             freq=1,
    #         )

    # # //////////////////////////////////////////////////////////////////////////////
    # # Grasps
    # # //////////////////////////////////////////////////////////////////////////////

    # # Pick and Place
    obj_id = 4
    assert robot.object_dicts[obj_id]["label"][0] == "bowl"

    robot.pick(obj_id)
