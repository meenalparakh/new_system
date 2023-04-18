import numpy as np
import random
from copy import copy
import cv2
import seaborn as sns


VOWELS = ['a', 'e', 'i', 'o', 'u']
count_prefixes = ["first", "second", "third", "fourth", "fifth"]


def arrange_edges(object_dicts, new_obj_lst):
    for obj_id in new_obj_lst:
        info = object_dicts[obj_id]
        edges = list(info["edges"])
        centers = [object_dicts[e]["object_center"] for e in edges]
        dist = list(
                np.linalg.norm(np.array(centers) - np.array(info["object_center"]).reshape((1,3)), 
                                   axis=1)
                )
        info["edges"] = [e for _, e in sorted(zip(dist, edges))]
        print(obj_id, info["edges"], sorted(dist))

def capitalize_first_letter(sentence):
    return sentence[:1].upper() + sentence[1:]

def add_prefix(object_name, lst, counts, idx=None, prefix=None):
    if idx is None:
        i = lst.index(object_name)
    else:
        i = idx
    if counts[i] == 1:
        if prefix == "the":
            name = "the " + object_name
        elif prefix == "default":
            name = obj_fn1(object_name)
        else:
            name = object_name
            # name = object_name
    else:
        name = str(counts[i]) + " " + object_name + "s"
    return name

def join_objects(objects_name_lst, return_counts=False, prefix=None):
    if len(objects_name_lst) == 0:
        combined = ''
        if return_counts:
            return combined, [], []
        else:
            return combined

    lst = []
    counts = []
    for obj_name in objects_name_lst:
        if obj_name in lst:
            continue
        lst.append(obj_name)
        counts.append(objects_name_lst.count(obj_name))

    description_lst = []
    for i in range(len(lst)):
        description_lst.append(add_prefix(lst[i], lst, counts, i, prefix=prefix))

    if len(lst) == 1:
        combined = description_lst[0]
    elif len(lst) == 2:
        combined = " and ".join(description_lst)
    else:
        print("description_lst", description_lst)
        description_lst[-1] = "and " + description_lst[-1]
        combined = ", ".join(description_lst)

    if return_counts:
        return combined, lst, counts
    return combined

def obj_fn1(obj_name):
    if obj_name[0] in VOWELS: 
        return "an " + obj_name
    return "a " + obj_name

def contains_template(objects_name_lst, container_name, voice="active"):
    if len(objects_name_lst) == 0:
        return ''
    if voice == 'active':
        # start = " The " + container_name
        start = " " + container_name
        start = start[:2].upper() + start[2:] + " contains "
        return start + join_objects(objects_name_lst, prefix=None) + "."
    else:
        start = join_objects(objects_name_lst, prefix=None)
        start = start[:1].upper() + start[1:]
        identifier = "is" if len(objects_name_lst) == 1 else "are"
        possibles = [" " + start + f" {identifier} in " + (container_name) + ".",
                     " " + start + (" lies " if identifier == "is" else " lie ") + "in " \
                            + f"{container_name}."
                    ]
        return random.choice(possibles)

def placed_over_template(above_obj_lst, below_obj, voice="active"):
    if len(above_obj_lst) == 0:
        return ''
    if voice == 'active':
        start = " Over " + below_obj
        start = start[:1].upper() + start[1:] + " lies "
        return start + join_objects(above_obj_lst, prefix=None) + "."
    else:
        start = join_objects(above_obj_lst, prefix=None)
        start = start[:1].upper() + start[1:]
        identifier = "lies" if len(above_obj_lst) == 1 else "lie"
        possibles = " " + start + f" {identifier} over " + (below_obj) + "."
        return possibles

def draw_graph(object_dicts, path, size):
    h, w = size
    graph = 255*np.ones((h, w, 3), dtype=np.uint8)
    for j in object_dicts.keys():
        pix = object_dicts[j]["pixel_center"]
        # a, b, c, d = get_bounding_box(pix, 10, graph.shape)
        graph = cv2.circle(graph, pix, radius=2, color=(36,36,255), thickness=-1)
        cv2.putText(graph, object_dicts[j]["name"], (pix[0]+10, pix[1]+10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1)
        for obj_id in list(object_dicts[j]["contains"]):
            end_point = object_dicts[obj_id]["pixel_center"]
            cv2.line(graph, pix, end_point, (0, 0, 200), 1) 

    num_edges = len(path)
    colors = sns.color_palette("rocket", num_edges)
    for edge in path:
        start_point = object_dicts[edge[0]]["pixel_center"]
        end_point = object_dicts[edge[1]]["pixel_center"]
        color_val = colors.pop(0)
        color = tuple([int(p*255) for p in color_val])
        cv2.line(graph, start_point, end_point, color, 2) 

    cv2.imwrite('./graph.png', graph)

def graph_traversal(object_dicts, new_obj_lst, side="right"):
    start = [object_dicts[o]["object_center"][1] for o in new_obj_lst]
    # start = [np.mean(object_dicts[o]["pcd"], axis=0)[1] for o in new_obj_lst]

    start = new_obj_lst[np.argmin(start)] if side == 'left' else new_obj_lst[np.argmax(start)]
    print(f"Starting node is {start}")
    remaining = set(new_obj_lst)
    travelled = [start]
    path = []
    branches = [(start, e) for e in object_dicts[start]["edges"]]
    remaining.remove(start)
    while len(remaining) > 0:
        while len(branches) != 0:
            source, chosen_obj = branches.pop(0)
            if chosen_obj in travelled:
                continue
            path.append((source, chosen_obj))
            travelled.append(chosen_obj)
            remaining.remove(chosen_obj)
            branches = [(chosen_obj, e) for e in object_dicts[chosen_obj]["edges"]] + branches
        if len(remaining) > 0:
            remaining_ids = list(remaining)
            centers = np.array([object_dicts[id]["object_center"] for id in remaining_ids])
            last_center = np.array(object_dicts[travelled[-1]]["object_center"]).reshape((1, 3))
            next_id = remaining_ids[np.argmin(np.linalg.norm(centers-last_center, axis=1))]
            branches = [(travelled[-1], next_id)]
            object_dicts[travelled[-1]]["edges"].append(next_id)
            object_dicts[next_id]["edges"].append(travelled[-1])

    print("list for travelled and the new object list", travelled, new_obj_lst)
    assert len(travelled)==len(new_obj_lst)
    return travelled, path

def construct_graph(object_dict, threshold_dist=0.3):

    to_remove = []
    obj_ids = list(object_dict.keys())
    for o1 in obj_ids:
        # if "relation" in object_dict[o1] and "contains" in object_dict[o1]["relation"]:
        to_remove.extend(object_dict[o1]["relation"]["contains"])
        # if "relation" in object_dict[o1] and "below" in object_dict[o1]["relation"]:
        to_remove.extend(object_dict[o1]["relation"]["below"])

        object_dict[o1]["object_center"] = np.median(object_dict[o1]["pcd"], axis=0)
            
    reduced_obj_lst = list(set(obj_ids).difference(set(to_remove)))
    reduced_obj_array = np.array(reduced_obj_lst)

    center_lst = [np.mean(object_dict[o1]["pcd"], axis=0) for o1 in reduced_obj_lst]
    center_lst = np.array(center_lst)

    for o1 in reduced_obj_lst:
        object_dict[o1]["edges"] = set()

    for o1_idx, o1 in enumerate(reduced_obj_lst):
        
        center = center_lst[o1_idx:o1_idx+1, :]
        ds = np.linalg.norm(center_lst - center, axis=1)
        ds[o1_idx] = 100
        k = min(1, len(ds)-1)
        indices = np.argpartition(ds, k)
        if (len(ds) > 1) and (ds[indices[1]] > threshold_dist):
            to_add = reduced_obj_array[indices[0]]
            object_dict[o1]["edges"].add(to_add)
            object_dict[to_add]["edges"].add(o1)
        else:
            to_add1, to_add2 = list(reduced_obj_array[indices[:2]])
            object_dict[o1]["edges"].update([to_add1, to_add2])
            object_dict[to_add1]["edges"].add(o1)
            object_dict[to_add2]["edges"].add(o1)

    # print(object_dict)
    print("reduced obj lst", reduced_obj_lst)
    arrange_edges(object_dict, reduced_obj_lst)
    print("Arranged edges")
    return object_dict, reduced_obj_lst

def get_object_label_wrt_table(direction_vector):
    '''
    direction_vector: normalized direction from the origin of the table.
    1, 0 represents the mid of the right side.
    0, 1 represents the mid of the top side.

    '''
    lst = {
        0: f"close to the table's left edge",
        1: f"close to the table's right edge",
        2: f"close to the table's top edge",
        3: f"close to the table's bottom edge",
        4: f"close to the edge of the table",
        5: "near the center of the left edge",
        6: "near the center of the right edge",
        7: "near the center of the top edge",
        8: "near the center of the bottom edge",
        9: "near the top right corner",
        10: "near the top left corner",
        11: "near the bottom right corner",
        12: "near the bottom left corner",
        13: "close to table's center"
    }
    return [lst[0]]


def get_direction_label(direction_vector, obj_name):
    lst = {
        0: f"very close to {obj_name}", 
        1: f"to the right of {obj_name}",
        2: f"to {obj_name}'s right side",
        3: f"to the left of {obj_name}",
        4: f"to {obj_name}'s left side",
        5: f"a little down from {obj_name}",
        6: f"a little ahead of {obj_name}",
        7: f"further away to the right of {obj_name}",
        8: f"further away to the left of {obj_name}",
        9: f"further down from {obj_name}",
        10: f"further ahead of {obj_name}",
        11: f"to {obj_name}'s diagonal right",
        12: f"to the diagonal right of {obj_name}",         
        13: f"to {obj_name}'s diagonal left",
        14: f"to the diagonal left of {obj_name}",
        15: f"farther away from {obj_name}",
        16: f"near to {obj_name}"
    }

    dist = np.linalg.norm(direction_vector)
    direction = np.arctan2(direction_vector[1:2], direction_vector[0:1])[0]
    side = "left" if direction < 0 else "right"
    precise_description = f"{np.abs(direction)*180/np.pi} degrees from the vertical," + \
            f" to the {side}, at a distance of {dist*100} cm"

    if  dist < 0.05:
        return [lst[0]]
    elif dist > 0.20:
        dist_label = "farther"
    else:
        dist_label = "some distance away"

    if np.abs(direction) < np.pi/12:
        direction_label = "down"
        if dist_label == "farther":
            return [lst[9]]
        return [lst[5]] 
    elif np.abs(direction - np.pi/4) < np.pi/6:
        direction_label = "bottom right diagonal"; return [lst[11], lst[12]]
    elif np.abs(direction - np.pi/2) < np.pi/12:
        direction_label = "right side"
        if dist_label == "farther":
            return [lst[7]]
        return [lst[1], lst[2]]
    elif np.abs(direction - 3*np.pi/4) < np.pi/6:
        direction_label = "top right diagonal"; return [lst[11], lst[12]]
    elif np.abs(direction + np.pi/4) < np.pi/6:
        direction_label = "bottom left diagonal"; return [lst[13], lst[14]]
    elif np.abs(direction + np.pi/2) < np.pi/12:
        direction_label = "left side" 
        if dist_label == "farther":
            return [lst[8]]
        return [lst[3], lst[4]]
    elif np.abs(direction + 3*np.pi/4) < np.pi/6:
        direction_label = "top left diagonal"; return [lst[13], lst[14]]
    elif (np.pi - np.abs(direction)) < np.pi/12:
        direction_label = "up"
        if dist_label == "farther":
            return [lst[10]]
        return [lst[6]]
    else:
        # direction_label = "diagonal"
        result = [lst[15]] if dist_label == "farther" else [lst[16]]
        return result


def text_description(object_dicts, traversal_order, path, side="right"):
    # traversal_order, path = graph_traversal(object_dicts, new_obj_lst, side=side)
    path = copy(path)
    object_names = []
    for obj in traversal_order:
        # dic = object_dicts[obj]
        # object_names.append(dic["name"])
        # object_names.extend([object_dicts[c]["name"] for c in list(dic["contains"])])
        # object_names.extend([object_dicts[c]["name"] for c in list(dic["lies_below"])])

        obj_name = [object_dicts[obj]["label"][0]]
        # if "relation" in object_dicts[obj]:
        contained_ids = object_dicts[obj]["relation"].get("contains", [])
        above_ids = object_dicts[obj]["relation"].get("below", [])

        contained_names = [object_dicts[i]["label"][0] for i in contained_ids]
        obj_name.extend(contained_names)

        above_names = [object_dicts[i]["label"][0] for i in above_ids]
        obj_name.extend(above_names)

        object_names.extend(obj_name)
    
    print(object_names)

    combined, name_lst, counts = join_objects(object_names, return_counts=True, prefix="default")
    current_count = {name: 0 for name in name_lst}
    for obj_id in traversal_order:
        name = object_dicts[obj_id]["label"][0]
        print("P name:", name)
        count = counts[name_lst.index(name)]
        if count == 1:
            object_dicts[obj_id]["used_name"] = "the " + name
        else:
            object_dicts[obj_id]["used_name"] = "the " + count_prefixes[current_count[name]] + \
                " " + name
            current_count[name] += 1

        contained_lst = object_dicts[obj_id]["relation"].get("contains", [])
        above_lst = object_dicts[obj_id]["relation"].get("below", [])
        for child_id in contained_lst + above_lst:
            name = object_dicts[child_id]["label"][0]
            print("C name:", name)
            count = counts[name_lst.index(name)]
            if count == 1:
                object_dicts[child_id]["used_name"] = "the " + name
            else:
                object_dicts[child_id]["used_name"] = "the " + count_prefixes[current_count[name]] + \
                    " " + name
                current_count[name] += 1

    for obj_id in object_dicts:
        print(object_dicts[obj_id]["label"])
        print(object_dicts[obj_id]["used_name"])

    possible_starts = [
        "On the table lies " + combined + ".",
        "A table has the following objects: " + combined + "."
    ]
    start = random.choice(possible_starts)
    first_object_dict = object_dicts[traversal_order.pop(0)]
    first_object_line = f" At the {side} of all the objects on the table lies " + \
                        first_object_dict["used_name"] + "." 

    
    contained_lst = first_object_dict["relation"]["contains"] 
    contained_object_names = [object_dicts[n]["used_name"] for n in contained_lst]
    contained_line = contains_template(contained_object_names, 
                first_object_dict["used_name"], voice="active")

    above_lst = first_object_dict["relation"]["below"] 
    above_object_names = [object_dicts[n]["used_name"] for n in above_lst]
    above_line = placed_over_template(above_object_names, first_object_dict["used_name"])

    print("here: contained and above", contained_lst, above_lst)
    first_object_dict["desc"] = f"lies at the {side} of all the objects on the table"
    for i in list(contained_lst):
        print(i, object_dicts[i]["label"], "desc set")
        object_dicts[i]["desc"] = f"lies inside " + first_object_dict["used_name"]

    for i in list(above_lst):
        object_dicts[i]["desc"] = f"lies over " + first_object_dict["used_name"]

    start = start + first_object_line + contained_line + above_line
    while len(path) > 0:
        source_obj, next_object_id = path.pop(0)
        source_obj_dict = object_dicts[source_obj]
        next_object_dict = object_dicts[next_object_id]
        next_object_name = next_object_dict["used_name"]
        direction_vector = np.array(next_object_dict["object_center"]) - \
                    np.array(source_obj_dict["object_center"])
        labels = get_direction_label(direction_vector, source_obj_dict["used_name"])
        label = random.choice(labels)

        if len(source_obj_dict["relation"]["contains"]) > 0:
            label = "outside, and " + label

        possible_next_lines = [
            label + " lies " + next_object_name + ".",
            label + ", " + next_object_name + " has been placed.",
            next_object_name + " lies " + label + "."
        ]
        chosen_line = random.choice(possible_next_lines)
        contained_object_names = [object_dicts[n]["used_name"] for n in next_object_dict["relation"]["contains"]]
        contained_line = contains_template(contained_object_names, next_object_name, voice="active")
        
        above_object_names = [object_dicts[n]["used_name"] for n in list(next_object_dict["relation"]["below"])]
        above_line = placed_over_template(above_object_names, next_object_name)

        next_object_dict["desc"] = "lies " + label
        for i in list(next_object_dict["relation"]["contains"]):
            object_dicts[i]["desc"] = f"lies inside {next_object_name}"

        for i in list(next_object_dict["relation"]["below"]):
            object_dicts[i]["desc"] = f"lies over " + next_object_name

        start = start + " " + capitalize_first_letter(chosen_line) + contained_line + above_line
    return start

def make_options(pick_targets, place_targets, options_in_api_form=True, termination_string="done()"):
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    print("Considering", len(options), "options")
    return options

def get_place_description(object_dicts, obj_lst, place_center):
    centers = [object_dicts[obj_id]["object_center"] for obj_id in obj_lst]
    centers = np.array(centers)

    dist = centers[:, :2] - np.array([place_center[:2]])
    dist = np.linalg.norm(dist, axis=1)
    obj_id = obj_lst[np.argmin(dist)]

    obj_info = object_dicts[obj_id]
    obj_pixel_center = (obj_info["object_center"])
    direction_vector = place_center[:2] - obj_pixel_center[:2]
    labels = get_direction_label(direction_vector, obj_info["used_name"])
    label = random.choice(labels)
    return label

