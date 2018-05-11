# coding = utf-8
from config import *
import random, math, copy, json
import numpy as np


def random_pick(some_list, p_list, pick_num=1):
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


def resolve_context(some_list, arg_num):
    """
    Judge if the agent can resolve context
    :param some_list: entity list or attribute list
    :param arg_num: 1 or 2, the number of considered arguments. In qa or confirm, it's 1; in compare, it's 2.
    :return: True or False
    """
    assert arg_num in [1, 2], "step should be in [1,2]"
    if arg_num == 1:
        if len(some_list) > 2 and some_list[-1] == some_list[-2] == some_list[-3]:
            return True
        elif len(some_list) == 2 and some_list[-1] == some_list[-2]:
            return True
        else:
            return False
    else:
        if len(some_list) > 3 and some_list[-2] == some_list[-3] == some_list[-4]:
            return True
        elif len(some_list) == 3 and some_list[-2] == some_list[-3]:
            return True
        else:
            return False


def qa_generator(script, attr_list, entity_list, wording_list, p_list):
    """
    Get a qa dialog script based on dialog history
    :param script:
    :param attr_list:
    :param entity_list:
    :param wording_list:
    :param p_list:
    :return:
    """
    assert len(attr_list) == len(entity_list), "The len of attribute list and entity list should be equal!"
    assert len(wording_list) == len(p_list), "The len of wording list and probability list should be equal!"

    # obtain basic scene
    attr = attr_list[-1]
    entity = entity_list[-1]
    wording = random_pick(wording_list, p_list)
    scene_name = ' '.join(['qa', str(entity), attr, wording])
    scene_content = list()

    for turn in script[attr][wording]:
        if type(turn) is list:
            scene_content.append(random.choice(turn))
        else:
            scene_content.append(turn)

    # Now we make context important
    if wording == 'lack_entity' and resolve_context(entity_list, 1):
        scene_content = [scene_content[0], scene_content[-1]]
    elif wording == 'lack_attribute' and resolve_context(attr_list, 1):
        scene_content = [scene_content[0], scene_content[-1]]
    else:
        pass

    scene = {scene_name: scene_content}
    return scene


def confirm_generator(script, attr_list, entity_list, wording_list, p_list):
    """
    Get a confirm dialog script based on dialog history
    :param script:
    :param attr_list:
    :param entity_list:
    :param wording_list:
    :param p_list:
    :return:
    """
    assert len(attr_list) == len(entity_list), "The len of attribute list and entity list should be equal!"
    assert len(wording_list) == len(p_list), "The len of wording list and probability list should be equal!"

    # obtain basic scene
    attr = attr_list[-1]
    entity = entity_list[-1]
    wording = random_pick(wording_list, p_list)
    scene_name = ' '.join(['confirm', str(entity), attr, wording])
    scene_content = list()

    for turn in script[attr][wording]:
        if type(turn) is list:
            scene_content.append(random.choice(turn))
        else:
            scene_content.append(turn)

    # Now we make context important
    if wording == 'lack_entity' and resolve_context(entity_list, 1):
        scene_content = [scene_content[0], scene_content[-1]]
    else:
        pass

    scene = {scene_name: scene_content}
    return scene


def compare_generator(script, attr_list, entity_list, wording_list, p_list, move, step):
    """
    Get a confirm dialog script based on dialog history
    :param script:
    :param attr_list:
    :param entity_list:
    :param wording_list:
    :param p_list:
    :param step:
    :param move:
    :return:
    """
    assert len(attr_list) == len(entity_list), "The len of attribute list and entity list should be equal!"
    assert len(wording_list) == len(p_list), "The len of wording list and probability list should be equal!"
    assert attr_list[-1] == attr_list[-2], "We should compare the same attribute!"

    # obtain basic scene
    attr = attr_list[-1]
    entity1 = entity_list[-1]
    entity2 = entity_list[-2]
    wording = random_pick(wording_list, p_list)
    scene_name = ' '.join(['compare', str(entity1), str(entity2), attr, wording])
    scene_content = list()

    for turn in script[attr][wording]:
        if type(turn) is list:
            scene_content.append(random.choice(turn))
        else:
            scene_content.append(turn)

    # Now we make context important
    if step == 1:
        if move == 'diagonal':
            if wording == 'lack_entity' and resolve_context(entity_list, 2):
                scene_content = [scene_content[0], scene_content[-1]]
            else:
                pass
        elif move == 'horizontal':
            if wording == 'lack_entity' and resolve_context(entity_list, 2):
                scene_content = [scene_content[0], scene_content[-1]]
            elif wording == 'lack_attribute' and resolve_context(attr_list, 2):
                scene_content = [scene_content[0], scene_content[-1]]
            elif wording == 'lack_attribute_entity':
                if resolve_context(entity_list, 2) and resolve_context(attr_list, 2):
                    scene_content = [scene_content[0], scene_content[-1]]
                elif resolve_context(entity_list, 2) and not resolve_context(attr_list, 2):
                    scene_content = [scene_content[0], scene_content[1], scene_content[2], scene_content[-1]]
                elif not resolve_context(entity_list, 2) and resolve_context(attr_list, 2):
                    scene_content = [scene_content[0], scene_content[3], scene_content[4], scene_content[-1]]
                else:
                    pass
        else:
            assert True is False, "Unconsidered conditions happen!"
    else:
        if move == 'horizontal' and wording == 'lack_attribute' and resolve_context(attr_list, 2):
            scene_content = [scene_content[0], scene_content[3], scene_content[4], scene_content[-1]]
        else:
            pass

    scene = {scene_name: scene_content}
    return scene


def filter_p_dict(available_list, p_dict):
    new_p_dict = dict()
    p_sum = 0
    for item in available_list:
        new_p_dict[item] = p_dict[item]
        p_sum += p_dict[item]
    for key in new_p_dict.keys():
        new_p_dict[key] /= p_sum
    return new_p_dict


def coordinate_p(matrix, previous_coordinate):
    available_coordinate = list()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                available_coordinate.append((i, j))
    p_list = [1 / len(available_coordinate)] * len(available_coordinate)

    # User prefer to explore a attribute or a entity without interrupt
    i_spare = dict()
    j_spare = dict()
    for coordinate in available_coordinate:
        i_spare[coordinate[0]] = i_spare.get(coordinate[0], 0) + 1
        j_spare[coordinate[1]] = j_spare.get(coordinate[1], 0) + 1

    p_amplify_list = list()
    for coordinate in available_coordinate:
        i, j = coordinate
        if previous_coordinate is not None:
            bonus = 1 if i == previous_coordinate[0] or j == previous_coordinate[1] else 0
        else:
            bonus = 0
        amplify = math.exp((matrix.shape[1] - i_spare[i] + matrix.shape[0] - j_spare[j] + bonus) / 1.5)
        p_amplify_list.append(amplify)

    p_list = [p * multi for p, multi in zip(p_list, p_amplify_list)]

    # slightly sharp probability and normalize
    p_list = [math.exp(3.5 * p) for p in p_list]
    p_sum = sum(p_list)
    p_list = [p / p_sum for p in p_list]

    return available_coordinate, p_list


def calculate_move(previous_coordinate, *args):
    assert 0 < len(args) < 3, "Illegal args!"

    if previous_coordinate is None:
        return 'init'

    if len(args) == 2:
        step = 2
        current_coordinate_1 = args[0]
        current_coordinate_2 = args[1]
        assert current_coordinate_1[0] == current_coordinate_2[0], "This is only for compare!"
        assert current_coordinate_1[1] != current_coordinate_2[1], "This is only for compare!"

        if current_coordinate_1[0] == previous_coordinate[0]:
            action = 'horizontal'
        else:
            action = 'diagonal'
    else:
        step = 1
        current_coordinate = args[0]
        if current_coordinate[0] == previous_coordinate[0]:
            action = 'horizontal'
        elif current_coordinate[1] == previous_coordinate[1]:
            action = 'vertical'
        else:
            action = 'diagonal'

    return "_".join([action, str(step)])


def get_compare_entity(available_coordinate, p_list, entity_num):
    # calculate probability sum for each row and sharp it
    row_p_dict = dict()
    for coordinate, p in zip(available_coordinate, p_list):
        i = coordinate[0]
        row_p_dict[i] = row_p_dict.get(i, 0) + p
    for key, value in row_p_dict.items():
        row_p_dict[key] = math.exp(5 * value)

    # choose which row to compare and get probability for each coordinate in that row and sharp it
    row = random_pick(list(row_p_dict.keys()), list(row_p_dict.values()))
    available_coordinate_in_row = list()
    p_list_in_row = list()
    for coordinate, p in zip(available_coordinate, p_list):
        if coordinate[0] == row:
            available_coordinate_in_row.append(coordinate)
            p_list_in_row.append(p)
    p_list_in_row = [math.exp(5 * p) for p in p_list_in_row]

    # choose the entity to compare and how many entity
    if len(available_coordinate_in_row) > 1 and random.random() < 0.5:
        coordinate1, coordinate2 = random_pick(available_coordinate_in_row, p_list_in_row, 2)
        explored_num = 2
    else:
        coordinate1 = random_pick(available_coordinate_in_row, p_list_in_row)
        explored_num = 1
        all_entity = list(range(entity_num))
        for coordinate in available_coordinate_in_row:
            all_entity.remove(coordinate[1])
        coordinate2_j = random_pick(all_entity, [1] * len(all_entity))
        coordinate2 = (row, coordinate2_j)
    return coordinate1, coordinate2, explored_num


def pre_sales_controller(script, user_concern_attr, user_concern_entity, available_intent, grammar_p_dict,
                         intent_p_dict):
    """
    Control the flow of pre_sales.
    :param script: all dialog script in pre_sales
    :param user_concern_attr: a sampled attributes list which users may concern
    :param user_concern_entity: a sampled entities list which users may concern
    :param available_intent: available pre_sales intents list in current considered dialog complicity
    :param grammar_p_dict: GRAMMAR_P_DICT when keys == 'pre_sales'
    :param intent_p_dict: INTENT_P_DICT when keys == 'pre_sales'
    :return:
    """
    matrix = np.zeros((len(user_concern_attr), len(user_concern_entity)), dtype=int)
    attr_list = list()
    entity_list = list()
    terminal = False
    pre_sales_script = list()
    previous_coordinate = None

    while not terminal:
        available_intent_p_dict = filter_p_dict(available_intent, intent_p_dict)
        intent = random_pick(list(available_intent_p_dict.keys()), list(available_intent_p_dict.values()))
        available_script = script[intent]
        if intent == 'qa' or intent == 'confirm':
            available_coordinate, p_list = coordinate_p(matrix, previous_coordinate)
            coordinate = random_pick(available_coordinate, p_list)

            move = calculate_move(previous_coordinate, coordinate)
            matrix[coordinate] = 1
            available_grammar_p_dict = grammar_p_dict[intent][move]

            attr_list.append(user_concern_attr(coordinate[0]))
            entity_list.append(user_concern_entity(coordinate[1]))

            if intent == 'qa':
                scene = qa_generator(available_script, attr_list, entity_list, list(available_grammar_p_dict.keys()),
                                     list(available_grammar_p_dict.values()))
            else:
                scene = confirm_generator(available_script, attr_list, entity_list,
                                          list(available_grammar_p_dict.keys()),
                                          list(available_grammar_p_dict.values()))
            previous_coordinate = coordinate
        else:
            available_coordinate, p_list = coordinate_p(matrix, previous_coordinate)
            coordinate1, coordinate2, explored_num = get_compare_entity(available_coordinate, p_list, matrix.shape[1])

            if explored_num == 1:
                move = calculate_move(previous_coordinate, coordinate1)
                matrix[coordinate1] = 1
            else:
                move = calculate_move(previous_coordinate, coordinate1, coordinate2)
                matrix[coordinate1] = matrix[coordinate2] = 1
            available_grammar_p_dict = grammar_p_dict[intent][move]

            attr_list.append(user_concern_attr(coordinate2[0]))
            attr_list.append(user_concern_attr(coordinate1[0]))
            entity_list.append(user_concern_entity(coordinate2[1]))
            entity_list.append(user_concern_entity(coordinate1[1]))

            scene = compare_generator(available_script, attr_list, entity_list, list(available_grammar_p_dict.keys()),
                                      list(available_grammar_p_dict.values()), move, explored_num)
            previous_coordinate = coordinate1

        pre_sales_script.append(scene)
        if np.sum(matrix) == matrix.shape[0] * matrix.shape[1]:
            terminal = True

    return pre_sales_script


if __name__ == "__main__":
    script_f = DATA_ROOT + 'script.txt'
    with open(script_f, 'rb') as f:
        script = json.load(f)
    complicity = AVAILABLE_INTENT_1
    script = script['pre_sales']
    user_concern_attr = ['price', 'nfc']
