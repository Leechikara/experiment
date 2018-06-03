# coding=utf-8
import copy, random


def filter_p_dict(available_list, p_dict):
    """
    keep the items of p_dict where key in available_list and delete other items
    """
    new_p_dict = dict()
    p_sum = 0
    for item in available_list:
        if item in p_dict.keys():
            new_p_dict[item] = p_dict[item]
            p_sum += p_dict[item]
    for key in new_p_dict.keys():
        new_p_dict[key] /= p_sum
    return new_p_dict


def random_pick(some_list, p_list, pick_num=1):
    """
    Pick a few items from some_list according the weight declared by p_list
    """
    assert len(some_list) >= pick_num
    assert len(some_list) == len(p_list)

    some_list_copy = copy.deepcopy(some_list)
    p_list_copy = copy.deepcopy(p_list)
    picked_item = list()

    for _ in range(pick_num):
        # normalization
        p_sum = sum(p_list_copy)
        p_list_copy = [x / p_sum for x in p_list_copy]

        p = random.random()
        cumulative_probability = 0
        for i, (item, item_p) in enumerate(zip(some_list_copy, p_list_copy)):
            cumulative_probability += item_p
            if p < cumulative_probability:
                picked_item.append(item)
                del p_list_copy[i]
                del some_list_copy[i]
                break
    if pick_num == 1:
        return picked_item[0]
    else:
        return picked_item


def find_attr_entity(scene_element):
    """
    Given scene_element describing which topic is, find which one is entity and which one is attr.
    """
    if type(scene_element) == str:
        scene_element = scene_element.split(" ")
    assert type(scene_element) == list

    if "compare" in scene_element:
        entity = list()
    else:
        entity = None
    attr = None

    for ele in scene_element:
        if ele.find("attr=") != -1:
            attr = ele.replace("attr=", "")
        if ele.find("entityId=") != -1:
            if type(entity) != list:
                entity = int(ele.replace("entityId=", ""))
            else:
                entity.append(int(ele.replace("entityId=", "")))

    if type(entity) == list:
        assert len(entity) == 2
    else:
        assert entity is not None

    if attr is None and "discountURL" in scene_element:
        attr = "discountURL"
    assert attr is not None

    return entity, attr
