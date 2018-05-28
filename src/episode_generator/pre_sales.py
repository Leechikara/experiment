# coding = utf-8
from config import *
import random, math, copy, json, os
import numpy as np
from collections import defaultdict
from utils import random_pick, filter_p_dict


class PreSales(object):
    def __init__(self, script, available_intent, intent_p_dict, grammar_p_dict):
        self.script = script
        self.available_intent = available_intent
        self.intent_p_dict = intent_p_dict
        self.grammar_p_dict = grammar_p_dict

        self.user_concern_attr = None
        self.user_concern_entity = None
        self.matrix = None
        self.attr_list = None
        self.entity_list = None
        self.intent_coordinate_dict = None
        self.history_intent = None
        self.previous_coordinate1 = None
        self.previous_coordinate2 = None
        self.coordinate = None
        self.intent = None
        self.episode_script = None

    def init_episode(self, user_concern_attr, user_concern_entity):
        self.user_concern_attr = user_concern_attr
        self.user_concern_entity = user_concern_entity
        self.matrix = np.zeros((len(user_concern_attr), len(user_concern_entity)), dtype=int)
        self.attr_list = list()
        self.entity_list = list()
        self.intent_coordinate_dict = defaultdict(list)
        self.history_intent = list()
        self.previous_coordinate1 = None
        self.previous_coordinate2 = None
        self.coordinate = None
        self.intent = None
        self.episode_script = list()

    def resolve_context(self, some_list, axis, user_concern_list):
        """
        Judge if the agent can resolve context and don't request users
        """
        assert self.intent in ['qa', 'confirm', 'compare']
        intent_size = len(self.history_intent)
        if intent_size > 2:
            concern_intent = self.history_intent[-3:]
        elif intent_size == 2:
            concern_intent = self.history_intent[0:]
        else:
            return False
        if len(concern_intent) == 3:
            if 'compare' in concern_intent and 'compare' != concern_intent[-1] or 'compare' not in concern_intent:
                del concern_intent[0]
            if 'compare' == concern_intent[-1] and 'compare' not in concern_intent[:-1] and intent_size > 3:
                concern_intent.insert(0, self.history_intent[-4])

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

        if self.intent in ['qa', 'confirm']:
            target = concern_some_list[-1]
            for item, intent in zip(concern_some_list[:-1], concern_intent[:-1]):
                if intent == 'compare':
                    if item[0] != item[1]:
                        return False
                    else:
                        item = item[0]
                if target != item:
                    if axis == 0 and self.matrix[self.coordinate[axis], user_concern_list.index(item)] != 1:
                        return False
                    if axis == 1 and self.matrix[user_concern_list.index(item), self.coordinate[axis]] != 1:
                        return False
            return True
        else:
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

    def coordinate_p(self):
        """
        Get coordinate probability in the matrix
        """
        available_coordinate = list()
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] == 0:
                    available_coordinate.append((i, j))
        p_list = [1 / len(available_coordinate)] * len(available_coordinate)

        # User prefer to explore a attribute or a entity without interrupt (Topic should be coherent.)
        i_spare = dict()
        j_spare = dict()
        for coordinate in available_coordinate:
            i_spare[coordinate[0]] = i_spare.get(coordinate[0], 0) + 1
            j_spare[coordinate[1]] = j_spare.get(coordinate[1], 0) + 1

        p_amplify_list = list()
        for coordinate in available_coordinate:
            i, j = coordinate
            if self.previous_coordinate1 is not None:
                bonus1 = 1 if i == self.previous_coordinate1[0] or j == self.previous_coordinate1[1] else 0
            else:
                bonus1 = 0
            if self.previous_coordinate2 is not None:
                bonus2 = 1 if i == self.previous_coordinate2[0] or j == self.previous_coordinate2[1] else 0
            else:
                bonus2 = 0
            amplify = math.exp(
                (self.matrix.shape[1] - i_spare[i] + self.matrix.shape[0] - j_spare[j] + bonus1 * 2 + bonus2) / 2)
            p_amplify_list.append(amplify)

        p_list = [p * multi for p, multi in zip(p_list, p_amplify_list)]

        # slightly sharp probability and normalize
        p_list = [math.exp(3 * p) for p in p_list]
        p_sum = sum(p_list)
        p_list = [p / p_sum for p in p_list]

        return available_coordinate, p_list

    def calculate_move(self, compared_coordinate=None):
        """
        In witch style the user explores the candidate entities in the matrix
        """
        if self.previous_coordinate1 is None:
            return 'init'

        if compared_coordinate is not None:
            step = 2
            assert self.coordinate[0] == compared_coordinate[0], "This is only for compare!"
            assert self.coordinate[1] != compared_coordinate[1], "This is only for compare!"

            if self.coordinate[0] == self.previous_coordinate1[0]:
                action = 'horizontal'
            else:
                action = 'diagonal'
        else:
            step = 1
            if self.intent in ['qa', 'confirm']:
                if self.coordinate[0] == self.previous_coordinate1[0]:
                    action = 'horizontal'
                elif self.coordinate[1] == self.previous_coordinate1[1]:
                    action = 'vertical'
                else:
                    action = 'diagonal'
            else:
                if self.coordinate[0] == self.previous_coordinate1[0]:
                    action = 'horizontal'
                else:
                    action = 'diagonal'

        return "_".join([action, str(step)])

    def get_compared_coordinate(self, available_coordinate, p_list):
        """
        In the compare scene, we choose which target to be compared
        """
        row = self.coordinate[0]
        available_coordinate_in_row = list()
        p_list_in_row = list()
        for coordinate, p in zip(available_coordinate, p_list):
            if coordinate[0] == row:
                available_coordinate_in_row.append(coordinate)
                p_list_in_row.append(p)
        p_list_in_row = [math.exp(5 * p) for p in p_list_in_row]

        # choose the entity to compare
        entity_num = len(self.user_concern_entity)
        all_entity = list(range(entity_num))
        for coordinate in available_coordinate_in_row:
            all_entity.remove(coordinate[1])
        all_entity.remove(self.coordinate[1])

        if len(all_entity) == 0 or len(available_coordinate_in_row) > 1 and random.random() < 0.5:
            explored_new = True
            compared_coordinate = random_pick(available_coordinate_in_row, p_list_in_row)
        else:
            explored_new = False
            compared_coordinate_j = random_pick(all_entity, [1] * len(all_entity))
            compared_coordinate = (row, compared_coordinate_j)

        return compared_coordinate, explored_new

    def qa_generator(self, available_script, wording_list, p_list, axis, concern_list):
        """
        Get a qa scene based on dialog history
        """
        assert len(self.attr_list) == len(self.entity_list)
        assert len(wording_list) == len(p_list)

        # obtain basic scene
        attr = self.attr_list[-1]
        entity = self.entity_list[-1]
        wording = random_pick(wording_list, p_list)
        scene_name = ' '.join(['qa', str(entity), attr, wording])
        scene_content = list()

        for turn in available_script[attr][wording]:
            if type(turn) is list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_content = [turn.replace('entity', str(entity)) for turn in scene_content]

        # Now we make context important
        if wording == 'lack_entity' and self.resolve_context(self.entity_list, axis, concern_list):
            scene_content = [scene_content[0], scene_content[-1]]
        elif wording == 'lack_attribute' and self.resolve_context(self.attr_list, axis, concern_list):
            scene_content = [scene_content[0], scene_content[-1]]
        else:
            pass

        scene = {scene_name: scene_content}
        return scene

    def confirm_generator(self, available_script, wording_list, p_list, axis, concern_list):
        """
        Get a confirm scene based on dialog history
        """
        assert len(self.attr_list) == len(self.entity_list)
        assert len(wording_list) == len(p_list)

        # obtain basic scene
        attr = self.attr_list[-1]
        entity = self.entity_list[-1]
        wording = random_pick(wording_list, p_list)
        scene_name = ' '.join(['confirm', str(entity), attr, wording])
        scene_content = list()

        for turn in available_script[attr][wording]:
            if type(turn) is list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_content = [turn.replace('entity', str(entity)) for turn in scene_content]

        # Now we make context important
        if wording == 'lack_entity' and self.resolve_context(self.entity_list, axis, concern_list):
            scene_content = [scene_content[0], scene_content[-1]]
        else:
            pass

        scene = {scene_name: scene_content}
        return scene

    def compare_generator(self, available_script, wording_list, p_list, axis, concern_list, move, explored_new):
        """
        Get a confirm scene based on dialog history
        """
        assert len(self.attr_list) == len(self.entity_list)
        assert len(wording_list) == len(p_list)
        assert self.attr_list[-1] == self.attr_list[-2], "We should compare the same attribute!"

        # obtain basic scene
        attr = self.attr_list[-1]
        entity1 = self.entity_list[-1]
        entity2 = self.entity_list[-2]
        wording = random_pick(wording_list, p_list)
        scene_name = ' '.join(['compare', str(entity1), str(entity2), attr, wording])
        scene_content = list()

        for turn in available_script[attr][wording]:
            if type(turn) is list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_content = [turn.replace('entity1', str(entity1)) for turn in scene_content]
        scene_content = [turn.replace('entity2', str(entity2)) for turn in scene_content]

        # Now we make context important
        if explored_new is False:
            if move == 'diagonal_1':
                if wording == 'lack_entity' and self.resolve_context(self.entity_list, axis, concern_list):
                    scene_content = [scene_content[0], scene_content[-1]]
                else:
                    pass
            elif move == 'horizontal_1':
                if wording == 'lack_entity' and self.resolve_context(self.entity_list, axis, concern_list):
                    scene_content = [scene_content[0], scene_content[-1]]
                elif wording == 'lack_attribute' and self.resolve_context(self.attr_list, axis, concern_list):
                    scene_content = [scene_content[0], scene_content[-1]]
                elif wording == 'lack_attribute_entity':
                    if self.resolve_context(self.entity_list, axis, concern_list) and self.resolve_context(
                            self.attr_list, axis, concern_list):
                        scene_content = [scene_content[0], scene_content[-1]]
                    elif self.resolve_context(self.entity_list, axis, concern_list) and not self.resolve_context(
                            self.attr_list, axis, concern_list):
                        scene_content = [scene_content[0], scene_content[1], scene_content[2], scene_content[-1]]
                    elif not self.resolve_context(self.entity_list, axis, concern_list) and self.resolve_context(
                            self.attr_list, axis, concern_list):
                        scene_content = [scene_content[0], scene_content[3], scene_content[4], scene_content[-1]]
                    else:
                        pass
            else:
                assert True is False, "Unconsidered conditions happen!"
        else:
            if move == 'horizontal_2' and wording == 'lack_attribute' and self.resolve_context(self.attr_list, axis,
                                                                                               concern_list):
                scene_content = [scene_content[0], scene_content[-1]]
            else:
                pass

        scene = {scene_name: scene_content}
        return scene

    def episode_generator(self):
        """
        Control the flow of pre_sales
        """
        terminal = False
        while not terminal:
            # First, we try to explore a new point
            available_coordinate, p_list = self.coordinate_p()
            self.coordinate = random_pick(available_coordinate, p_list)
            self.matrix[self.coordinate] = 1
            coordinate_index = available_coordinate.index(self.coordinate)
            del available_coordinate[coordinate_index]
            del p_list[coordinate_index]

            # Then we see if some intents are not feasible for current explored point
            # todo: if new attributes are added in KB, this part may need rewrite!
            current_available_intent = copy.deepcopy(self.available_intent)
            if self.user_concern_attr[self.coordinate[0]] not in COMPARE_PERMITTED_ATTR \
                    and 'compare' in current_available_intent:
                current_available_intent.remove('compare')
            if sum(self.matrix[self.coordinate[0]]) == 1 and 'confirm' in current_available_intent:
                current_available_intent.remove('confirm')
            if 'confirm' in current_available_intent:
                del_confirm = True
                for qa_coordinate in self.intent_coordinate_dict['qa']:
                    if self.coordinate[0] == qa_coordinate[0]:
                        del_confirm = False
                        break
                if del_confirm:
                    current_available_intent.remove('confirm')

            # Next we sample a intent and get according dialog script
            available_intent_p_dict = filter_p_dict(current_available_intent, self.intent_p_dict)
            self.intent = random_pick(list(available_intent_p_dict.keys()), list(available_intent_p_dict.values()))
            self.history_intent.append(self.intent)
            available_script = self.script[self.intent]
            self.intent_coordinate_dict[self.intent].append(self.coordinate)

            # generate specific script
            if self.intent in ['qa', 'confirm']:
                move = self.calculate_move()
                available_grammar_p_dict = copy.deepcopy(self.grammar_p_dict[self.intent][move])

                self.attr_list.append(self.user_concern_attr[self.coordinate[0]])
                self.entity_list.append(self.user_concern_entity[self.coordinate[1]])

                # Get topic target
                if move.find('horizontal') != -1:
                    axis = 1
                    concern_list = self.user_concern_attr
                elif move.find('vertical') != -1:
                    axis = 0
                    concern_list = self.user_concern_entity
                else:
                    axis = None
                    concern_list = None

                if self.intent == 'qa':
                    scene = self.qa_generator(available_script,
                                              list(available_grammar_p_dict.keys()),
                                              list(available_grammar_p_dict.values()),
                                              axis, concern_list)
                else:
                    scene = self.confirm_generator(available_script,
                                                   list(available_grammar_p_dict.keys()),
                                                   list(available_grammar_p_dict.values()),
                                                   axis, concern_list)
                self.previous_coordinate2 = self.previous_coordinate1
                self.previous_coordinate1 = self.coordinate
            else:
                compared_coordinate, explored_new = self.get_compared_coordinate(available_coordinate, p_list)

                if explored_new is False:
                    move = self.calculate_move()
                else:
                    move = self.calculate_move(compared_coordinate=compared_coordinate)
                    self.matrix[compared_coordinate] = 1

                available_grammar_p_dict = copy.deepcopy(self.grammar_p_dict[self.intent][move])
                available_grammar_p_dict = filter_p_dict(
                    list(available_script[self.user_concern_attr[self.coordinate[0]]].keys()),
                    available_grammar_p_dict)

                self.attr_list.append(self.user_concern_attr[compared_coordinate[0]])
                self.attr_list.append(self.user_concern_attr[self.coordinate[0]])
                self.entity_list.append(self.user_concern_entity[compared_coordinate[1]])
                self.entity_list.append(self.user_concern_entity[self.coordinate[1]])

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

                scene = self.compare_generator(available_script,
                                               list(available_grammar_p_dict.keys()),
                                               list(available_grammar_p_dict.values()),
                                               axis, concern_list, move, explored_new)
                self.previous_coordinate2 = self.previous_coordinate1
                self.previous_coordinate1 = self.coordinate

            self.episode_script.append(scene)
            if np.sum(self.matrix) == self.matrix.shape[0] * self.matrix.shape[1]:
                terminal = True

        return self.episode_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['scenes']['pre_sales']
    user_concern_attr = ['price', 'nfc', 'color', 'size']
    user_concern_entity = ['entityId=1', 'entityId=10', 'entityId=2', 'entityId=5']
    available_intent = AVAILABLE_INTENT_3['pre_sales']
    grammar_p_dict = GRAMMAR_P_DICT['pre_sales']
    intent_p_dict = INTENT_P_DICT['pre_sales']

    pre_sales = PreSales(script, available_intent, intent_p_dict, grammar_p_dict)

    # test our code
    random.seed(0)
    pre_sales.init_episode(user_concern_attr, user_concern_entity)
    pre_sales_script = pre_sales.episode_generator()
    for line in pre_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')

    print('.......................\n')

    random.seed(1)
    pre_sales.init_episode(user_concern_attr, user_concern_entity)
    pre_sales_script = pre_sales.episode_generator()
    for line in pre_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')

    print('.......................\n')

    random.seed(2)
    pre_sales.init_episode(user_concern_attr, user_concern_entity)
    pre_sales_script = pre_sales.episode_generator()
    for line in pre_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
