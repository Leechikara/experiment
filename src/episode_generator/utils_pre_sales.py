# coding = utf-8
from config import *
import random, math, copy, json, os
import numpy as np
from collections import defaultdict
from .utils import random_pick, filter_p_dict


def resolve_context(some_list, arg_num, matrix, coordinate, axis, user_concern_list, history_intent):
    """
    Judge if the agent can resolve context
    :param some_list: entity list or attribute list
    :param arg_num: 1 or 2, the number of considered arguments. In qa or confirm, it's 1; in compare, it's 2.
    :param matrix: attr entity matrix
    :param coordinate: current explored point
    :param axis: if horizontal, axis=1; if vertical, axis=0
    :param user_concern_list: a list of attr or entity
    :param history_intent: a list of attr or entity
    :return: True or False
    """
    assert arg_num in [1, 2], "step should be in [1,2]"
    intent_size = len(history_intent)
    if intent_size > 2:
        concern_intent = history_intent[-3:]
    elif intent_size == 2:
        concern_intent = history_intent[0:]
    else:
        return False
    if len(concern_intent) == 3:
        if 'compare' in concern_intent and 'compare' != concern_intent[-1] or 'compare' not in concern_intent:
            del concern_intent[0]
        if 'compare' == concern_intent[-1] and 'compare' not in concern_intent[:-1] and intent_size > 3:
            concern_intent.insert(0, history_intent[-4])

    position = -1
    concern_some_list = list()
    concern_intent.reverse()
    for intent in concern_intent:
        if intent == 'compare':
            concern_some_list.append((some_list[position - 1], some_list[position]))
            position -= 2
        else:
            concern_some_list.append(some_list[position])
            position -= 1
    concern_some_list.reverse()
    concern_intent.reverse()

    if arg_num == 1:
        # for qa and confirm
        target = concern_some_list[-1]
        for item, intent in zip(concern_some_list[:-1], concern_intent[:-1]):
            if intent == 'compare':
                if item[0] != item[1]:
                    return False
                else:
                    item = item[0]
            if target != item:
                if axis == 0 and matrix[coordinate[axis], user_concern_list.index(item)] != 1:
                    return False
                if axis == 1 and matrix[user_concern_list.index(item), coordinate[axis]] != 1:
                    return False
        return True
    else:
        # for compare
        candidate_set = [concern_some_list[-1][0]]
        for item in concern_some_list[:-1]:
            if type(item) == tuple:
                candidate_set.append(item[0])
                candidate_set.append(item[1])
            else:
                candidate_set.append(item)
        if concern_some_list[-1][0] not in candidate_set[1:]:
            return False

        candidate_set = set(candidate_set)
        if concern_some_list[-1][-1] in candidate_set:
            candidate_set.remove(concern_some_list[-1][-1])

        if len(candidate_set) == 1 or len(candidate_set) == 0:
            return True

        return False


def qa_generator(script, attr_list, entity_list, wording_list, p_list, matrix, coordinate, axis, concern_list,
                 history_intent):
    """
    Get a qa dialog script based on dialog history
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

    scene_content = [turn.replace('entity', str(entity)) for turn in scene_content]

    # Now we make context important
    if wording == 'lack_entity' and resolve_context(entity_list, 1, matrix, coordinate, axis, concern_list,
                                                    history_intent):
        scene_content = [scene_content[0], scene_content[-1]]
    elif wording == 'lack_attribute' and resolve_context(attr_list, 1, matrix, coordinate, axis, concern_list,
                                                         history_intent):
        scene_content = [scene_content[0], scene_content[-1]]
    else:
        pass

    scene = {scene_name: scene_content}
    return scene


def confirm_generator(script, attr_list, entity_list, wording_list, p_list, matrix, coordinate, axis, concern_list,
                      history_intent):
    """
    Get a confirm dialog script based on dialog history
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

    scene_content = [turn.replace('entity', str(entity)) for turn in scene_content]

    # Now we make context important
    if wording == 'lack_entity' and resolve_context(entity_list, 1, matrix, coordinate, axis, concern_list,
                                                    history_intent):
        scene_content = [scene_content[0], scene_content[-1]]
    else:
        pass

    scene = {scene_name: scene_content}
    return scene


def compare_generator(script, attr_list, entity_list, wording_list, p_list, move, step, matrix, coordinate, axis,
                      concern_list, history_intent):
    """
    Get a confirm dialog script based on dialog history
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

    scene_content = [turn.replace('entity1', str(entity1)) for turn in scene_content]
    scene_content = [turn.replace('entity2', str(entity2)) for turn in scene_content]

    # Now we make context important
    if step == 1:
        if move == 'diagonal_1':
            if wording == 'lack_entity' and resolve_context(entity_list, 2, matrix, coordinate, axis, concern_list,
                                                            history_intent):
                scene_content = [scene_content[0], scene_content[-1]]
            else:
                pass
        elif move == 'horizontal_1':
            if wording == 'lack_entity' and resolve_context(entity_list, 2, matrix, coordinate, axis, concern_list,
                                                            history_intent):
                scene_content = [scene_content[0], scene_content[-1]]
            elif wording == 'lack_attribute' and resolve_context(attr_list, 2, matrix, coordinate, axis, concern_list,
                                                                 history_intent):
                scene_content = [scene_content[0], scene_content[-1]]
            elif wording == 'lack_attribute_entity':
                if resolve_context(entity_list, 2, matrix, coordinate, axis, concern_list,
                                   history_intent) and resolve_context(attr_list, 2, matrix, coordinate, axis,
                                                                       concern_list, history_intent):
                    scene_content = [scene_content[0], scene_content[-1]]
                elif resolve_context(entity_list, 2, matrix, coordinate, axis, concern_list,
                                     history_intent) and not resolve_context(attr_list, 2, matrix, coordinate, axis,
                                                                             concern_list, history_intent):
                    scene_content = [scene_content[0], scene_content[1], scene_content[2], scene_content[-1]]
                elif not resolve_context(entity_list, 2, matrix, coordinate, axis, concern_list,
                                         history_intent) and resolve_context(attr_list, 2, matrix, coordinate, axis,
                                                                             concern_list, history_intent):
                    scene_content = [scene_content[0], scene_content[3], scene_content[4], scene_content[-1]]
                else:
                    pass
        else:
            assert True is False, "Unconsidered conditions happen!"
    else:
        if move == 'horizontal_2' and wording == 'lack_attribute' and resolve_context(attr_list, 2, matrix, coordinate,
                                                                                      axis, concern_list,
                                                                                      history_intent):
            scene_content = [scene_content[0], scene_content[-1]]
        else:
            pass

    scene = {scene_name: scene_content}
    return scene


def coordinate_p(matrix, previous_coordinate1, previous_coordinate2):
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
        if previous_coordinate1 is not None:
            bonus1 = 1 if i == previous_coordinate1[0] or j == previous_coordinate1[1] else 0
        else:
            bonus1 = 0
        if previous_coordinate2 is not None:
            bonus2 = 1 if i == previous_coordinate2[0] or j == previous_coordinate2[1] else 0
        else:
            bonus2 = 0
        amplify = math.exp((matrix.shape[1] - i_spare[i] + matrix.shape[0] - j_spare[j] + bonus1 * 2 + bonus2) / 2)
        p_amplify_list.append(amplify)

    p_list = [p * multi for p, multi in zip(p_list, p_amplify_list)]

    # slightly sharp probability and normalize
    p_list = [math.exp(2 * p) for p in p_list]
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


def get_compared_coordinate(available_coordinate, p_list, chosen_coordinate, entity_num):
    # get probability for each coordinate in compared row and sharp it
    row = chosen_coordinate[0]
    available_coordinate_in_row = list()
    p_list_in_row = list()
    for coordinate, p in zip(available_coordinate, p_list):
        if coordinate[0] == row:
            available_coordinate_in_row.append(coordinate)
            p_list_in_row.append(p)
    p_list_in_row = [math.exp(5 * p) for p in p_list_in_row]

    # choose the entity to compare and how many entity
    all_entity = list(range(entity_num))
    for coordinate in available_coordinate_in_row:
        all_entity.remove(coordinate[1])
    all_entity.remove(chosen_coordinate[1])

    if len(all_entity) == 0 or len(available_coordinate_in_row) > 1 and random.random() < 0.5:
        explored_new = True
        compared_coordinate = random_pick(available_coordinate_in_row, p_list_in_row)
    else:
        explored_new = False
        compared_coordinate_j = random_pick(all_entity, [1] * len(all_entity))
        compared_coordinate = (row, compared_coordinate_j)

    return compared_coordinate, explored_new


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
    previous_coordinate1 = None
    previous_coordinate2 = None
    intent_coordinate_dict = defaultdict(list)
    history_intent = list()

    while not terminal:
        # First, we try to explore a new point
        available_coordinate, p_list = coordinate_p(matrix, previous_coordinate1, previous_coordinate2)
        coordinate = random_pick(available_coordinate, p_list)
        matrix[coordinate] = 1
        coordinate_index = available_coordinate.index(coordinate)
        del available_coordinate[coordinate_index]
        del p_list[coordinate_index]

        # Then we see if some intent is not feasible for current explored point
        # todo: if new attribute is added, this part may need rewrite!
        current_available_intent = copy.deepcopy(available_intent)
        if user_concern_attr[coordinate[0]] not in COMPARE_PERMITTED_ATTR and 'compare' in current_available_intent:
            current_available_intent.remove('compare')
        if sum(matrix[coordinate[0]]) == 1 and 'confirm' in current_available_intent:
            current_available_intent.remove('confirm')
        if 'confirm' in current_available_intent:
            del_confirm = True
            for qa_coordinate in intent_coordinate_dict['qa']:
                if coordinate[0] == qa_coordinate[0]:
                    del_confirm = False
                    break
            if del_confirm:
                current_available_intent.remove('confirm')

        # sample a intent and get according dialog script
        available_intent_p_dict = filter_p_dict(current_available_intent, intent_p_dict)
        intent = random_pick(list(available_intent_p_dict.keys()), list(available_intent_p_dict.values()))
        history_intent.append(intent)
        available_script = script[intent]
        intent_coordinate_dict[intent].append(coordinate)

        # generate specific script
        if intent == 'qa' or intent == 'confirm':
            move = calculate_move(previous_coordinate1, coordinate)
            available_grammar_p_dict = copy.deepcopy(grammar_p_dict[intent][move])

            attr_list.append(user_concern_attr[coordinate[0]])
            entity_list.append(user_concern_entity[coordinate[1]])

            # for context resolve and grammar probability bias
            if move.find('horizontal') != -1:
                axis = 1
                concern_list = user_concern_attr
                for key, value in available_grammar_p_dict.items():
                    if key.find('attr') != -1:
                        available_grammar_p_dict[key] = value * 2
            elif move.find('vertical') != -1:
                axis = 0
                concern_list = user_concern_entity
                for key, value in available_grammar_p_dict.items():
                    if key.find('entity') != -1:
                        available_grammar_p_dict[key] = value * 2
            else:
                axis = None
                concern_list = None
                for key, value in available_grammar_p_dict.items():
                    if key.find('complete') != -1:
                        available_grammar_p_dict[key] = value * 2

            if intent == 'qa':
                scene = qa_generator(available_script, attr_list, entity_list, list(available_grammar_p_dict.keys()),
                                     list(available_grammar_p_dict.values()), matrix, coordinate, axis, concern_list,
                                     history_intent)
            else:
                scene = confirm_generator(available_script, attr_list, entity_list,
                                          list(available_grammar_p_dict.keys()),
                                          list(available_grammar_p_dict.values()), matrix, coordinate, axis,
                                          concern_list, history_intent)
            previous_coordinate2 = previous_coordinate1
            previous_coordinate1 = coordinate
        else:
            compared_coordinate, explored_new = get_compared_coordinate(available_coordinate, p_list, coordinate,
                                                                        matrix.shape[1])

            if explored_new is False:
                explored_num = 1
                move = calculate_move(previous_coordinate1, coordinate)
                if move == 'vertical_1':
                    move = 'diagonal_1'
            else:
                explored_num = 2
                move = calculate_move(previous_coordinate1, coordinate, compared_coordinate)
                matrix[compared_coordinate] = 1

            available_grammar_p_dict = copy.deepcopy(grammar_p_dict[intent][move])
            available_grammar_p_dict = filter_p_dict(list(available_script[user_concern_attr[coordinate[0]]].keys()),
                                                     available_grammar_p_dict)

            attr_list.append(user_concern_attr[compared_coordinate[0]])
            attr_list.append(user_concern_attr[coordinate[0]])
            entity_list.append(user_concern_entity[compared_coordinate[1]])
            entity_list.append(user_concern_entity[coordinate[1]])

            # for context resolve and grammar probability bias
            axis = None
            concern_list = None
            if move.find('horizontal') != -1:
                for key, value in available_grammar_p_dict.items():
                    if key.find('attr') != -1:
                        available_grammar_p_dict[key] = value * 2
            elif move.find('vertical') != -1:
                for key, value in available_grammar_p_dict.items():
                    if key.find('entity') != -1:
                        available_grammar_p_dict[key] = value * 2
            else:
                for key, value in available_grammar_p_dict.items():
                    if key.find('complete') != -1:
                        available_grammar_p_dict[key] = value * 2

            scene = compare_generator(available_script, attr_list, entity_list, list(available_grammar_p_dict.keys()),
                                      list(available_grammar_p_dict.values()), move, explored_num, matrix, coordinate,
                                      axis, concern_list, history_intent)
            previous_coordinate2 = previous_coordinate1
            previous_coordinate1 = coordinate

        pre_sales_script.append(scene)
        if np.sum(matrix) == matrix.shape[0] * matrix.shape[1]:
            terminal = True

    return pre_sales_script


if __name__ == "__main__":
    random.seed(0)
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['scenes']['pre_sales']
    user_concern_attr = ['price', 'nfc', 'color', 'size']
    user_concern_entity = ['entityId=1', 'entityId=10', 'entityId=2', 'entityId=5']
    available_intent = AVAILABLE_INTENT_3['pre_sales']
    grammar_p_dict = GRAMMAR_P_DICT['pre_sales']
    intent_p_dict = INTENT_P_DICT['pre_sales']

    pre_sales_script = pre_sales_controller(script, user_concern_attr, user_concern_entity, available_intent,
                                            grammar_p_dict, intent_p_dict)
    for line in pre_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
