import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import cv2
from scipy.spatial import Delaunay

# def in_hull(p, hull):
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)

#     return hull.find_simplex(p)>=0

# tested = np.random.rand(1000,3)
# cloud  = np.random.rand(1000,3)
# print(in_hull(tested,cloud))


def get_object_relation(object_dicts, concept=["above", "contained_in"]):
    """
    return a dictionary wherever the concept exist.
    {moving_obj: anchor_obj for concept(anchor_obj, moving_obj) == True}
    """
    pass


def above_deprecated(anchor_pcd, moving_pcd):
    """
    take all points on moving pcd that are higher than
    anchor pcds max height, and find if any of them (>0.01)
    lies inside the anchor_pcd mask

    if yes then two possiblities: above or contained
        if the ratio of points in moving pcd that lie above and
        below the anchor's highest point is very large - then
        said to be lying "above",
        if considerable part is inside that is ratio of higher
        and lower is atleast 0.4, it is "contained"
    it no then False
    """

    max_anchor_height = np.max(anchor_pcd[:, 2])
    pts_above_anchor = moving_pcd[moving_pcd[:, 2] > max_anchor_height]

    if len(pts_above_anchor) < 10:
        print("False. No points above anchor")
        return False

    anchor_hull = Delaunay(anchor_pcd[:, :2])
    pts_within_anchor = anchor_hull.find_simplex(moving_pcd[:, :2]) >= 0
    print(pts_within_anchor.shape, moving_pcd.shape)
    pts_overlapping_anchor = moving_pcd[pts_within_anchor]

    pts_over_anchor = pts_overlapping_anchor[
        (pts_overlapping_anchor[:, 2] > max_anchor_height)
    ]

    if len(pts_over_anchor) < 10:
        print("False. No points above anchor, that overlap with the convex hull")
        return False

    # print("pcd points lie above anchor. Checking whether contained holds.")
    num_pts_below = len(pts_overlapping_anchor) - len(pts_over_anchor)
    ratio = num_pts_below / len(pts_overlapping_anchor)
    print("Ratio of points inside:", ratio, num_pts_below, len(pts_overlapping_anchor))

    if num_pts_below < 5:
        print("True. Very few points overlapping points that lie inside anchor.")
        return True
    elif ratio > 0.25:
        print("False. Many points lie inside anchor")
        return False
    else:
        print(
            "True. By default. Number of pts above and below:",
            len(pts_overlapping_anchor) - num_pts_below,
            num_pts_below,
        )
        return True


def get_concept_label(
    anchor_pcd, moving_pcd, contain_threshold=0.2, above_pts_threshold=2
):
    result = np.zeros(13)

    pts_within_anchor = Delaunay(anchor_pcd[:, :2]).find_simplex(moving_pcd[:, :2]) >= 0
    moving_pts_overlapping = moving_pcd[pts_within_anchor]

    if len(moving_pts_overlapping) < above_pts_threshold:
        print("False. Number of overlapping points is small:", moving_pts_overlapping)
        overlapping = False
    else:
        overlapping = True

    if overlapping:
        pts_within_moving = (
            Delaunay(moving_pts_overlapping[:, :2]).find_simplex(anchor_pcd[:, :2]) >= 0
        )
        anchor_pts_overlapping = anchor_pcd[pts_within_moving]

        max_ht_anchor = np.max(anchor_pts_overlapping[:, 2])
        max_ht_moving = np.max(moving_pts_overlapping[:, 2])

        if max_ht_anchor > max_ht_moving:
            possibility = ["below", "contains"]
        else:
            possibility = ["above", "contained_in"]

        print("Possibilities", possibility)
        if "contained_in" in possibility:
            pts_inside_anchor = Delaunay(anchor_pcd).find_simplex(moving_pcd) >= 0
            num_pts_inside = np.sum(pts_inside_anchor)
            ratio = num_pts_inside / len(moving_pcd)
            print(
                f"Pts inside: {num_pts_inside}, total pts: {len(moving_pcd)}, Ratio: {ratio}"
            )

            if ratio > contain_threshold:
                print("More than 1/4th points lie inside anchor.")
                # return "contained_in"
                result[7] = 1.0
                return result
            else:
                result[3] = 1.0
                return result
        else:
            pts_inside_moving = Delaunay(moving_pcd).find_simplex(anchor_pcd) >= 0
            num_pts_inside = np.sum(pts_inside_moving)

            ratio = num_pts_inside / len(anchor_pcd)
            print(
                f"Pts inside: {num_pts_inside}, total pts: {len(anchor_pcd)}, Ratio: {ratio}"
            )

            if ratio > contain_threshold:
                print("More than 1/4th points lie inside moving.")
                result[8] = 1.0
                return result
            else:
                result[4] = 1.0
                return result

    else:
        return result


def get_relation(anchor_pcd, moving_pcd, contain_threshold=0.2, above_pts_threshold=2):
    pts_within_anchor = Delaunay(anchor_pcd[:, :2]).find_simplex(moving_pcd[:, :2]) >= 0
    moving_pts_overlapping = moving_pcd[pts_within_anchor]

    if len(moving_pts_overlapping) < above_pts_threshold:
        print("False. Number of overlapping points is small:", moving_pts_overlapping)
        overlapping = False
    else:
        overlapping = True

    if overlapping:
        pts_within_moving = (
            Delaunay(moving_pts_overlapping[:, :2]).find_simplex(anchor_pcd[:, :2]) >= 0
        )
        anchor_pts_overlapping = anchor_pcd[pts_within_moving]
        if len(anchor_pts_overlapping) == 0:
            possibility = ["contained_in", "above"]
        else:
            max_ht_anchor = np.max(anchor_pts_overlapping[:, 2])
            max_ht_moving = np.max(moving_pts_overlapping[:, 2])
            if max_ht_anchor > max_ht_moving:
                possibility = ["below", "contains"]
            else:
                possibility = ["above", "contained_in"]

        print("Possibilities", possibility)
        if "contained_in" in possibility:
            pts_inside_anchor = Delaunay(anchor_pcd).find_simplex(moving_pcd) >= 0
            num_pts_inside = np.sum(pts_inside_anchor)
            ratio = num_pts_inside / len(moving_pcd)
            print(
                f"Pts inside: {num_pts_inside}, total pts: {len(moving_pcd)}, Ratio: {ratio}"
            )

            if ratio > contain_threshold:
                print("More than 1/4th points lie inside anchor.")
                # return "contained_in"
                return "contained_in"
            else:
                return "above"
        else:
            pts_inside_moving = Delaunay(moving_pcd).find_simplex(anchor_pcd) >= 0
            num_pts_inside = np.sum(pts_inside_moving)

            ratio = num_pts_inside / len(anchor_pcd)
            print(
                f"Pts inside: {num_pts_inside}, total pts: {len(anchor_pcd)}, Ratio: {ratio}"
            )

            if ratio > contain_threshold:
                print("More than 1/4th points lie inside moving.")
                return "contains"
            else:
                return "below"

    else:
        return None


def contain(anchor_pcd, moving_pcd):
    """
    if a reasonable part (atleast one-fourth) of moving_pcd that lies within the mask of anchor,
    is lower than anchor_pcd's highest point
    """
    max_anchor_height = np.max(anchor_pcd[:, 2])

    anchor_hull = Delaunay(anchor_pcd[:, :2])
    pts_within_anchor = anchor_hull.find_simplex(moving_pcd[:, :2]) >= 0
    print(pts_within_anchor.shape, moving_pcd.shape)
    pts_overlapping_anchor = moving_pcd[pts_within_anchor]

    num_pts_below = np.sum(pts_overlapping_anchor[:, 2] < max_anchor_height)
    total_pts = len(pts_overlapping_anchor)

    if num_pts_below > 100:
        print("True. Many (> 100) points lie inside the anchor")
        return True
    elif num_pts_below / total_pts > 0.25:
        return True
    else:
        return False


def aligned_horizontally(anchor_pcd, moving_pcd, state):
    """
    the 3 pca components, if there is a high difference then the one direction
    if the longest, we compare this with the other, along the horizontal direction
    """
    pass


def aligned_vaertically(anchor_pcd, moving_pcd, state):
    """
    the 3 pca components, if there is a high difference then the one direction
    if the longest, we compare this with the other, along the vertical direction.
    """
    pass


def near(anchor_pcd, moving_pcd, state):
    """
    ratio of distance between centers and the object sizes
    """
    pass


def left(anchor_pcd, moving_pcd, state):
    """
    use the maximum density of points, and check if it is left or right
    """
    pass


def right(anchor_pcd, moving_pcd, state):
    """
    use the maximum density of points, and check if it is left or right
    """
    pass


def front(anchor_pcd, moving_pcd, state):
    """
    use the maximum density of points, and check if it is left or right
    """
    pass


def back(anchor_pcd, moving_pcd, state):
    """
    use the maximum density of points, and check if it is left or right
    """
    pass
