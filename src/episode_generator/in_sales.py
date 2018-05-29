# coding=utf-8
import random, os, json
from utils import random_pick, filter_p_dict
from config import *


class InSales(object):
    def __init__(self, script, available_intent, intent_p_dict, grammar_p_dict):
        self.script = script
        self.available_intent = available_intent
        self.intent_p_dict = intent_p_dict
        self.available_intent_p_dict = filter_p_dict(self.available_intent, self.intent_p_dict)
        self.grammar_p_dict = grammar_p_dict

        self.desired_entity = None
        self.given_entity = None
        self.picked_intents = None
        self.episode_script = None

    def init_episode(self, desired_entity):
        self.desired_entity = desired_entity
        self.given_entity = False

        # Sample basic intents
        pick_num = random.randint(1, len(self.available_intent))
        picked_intents = random_pick(list(self.available_intent_p_dict.keys()),
                                     list(self.available_intent_p_dict.values()),
                                     pick_num)
        if type(picked_intents) is not list:
            self.picked_intents = [picked_intents]
        else:
            self.picked_intents = picked_intents

        self.episode_script = list()

    def scene_generator(self, intent):
        current_grammar_p_dict = self.grammar_p_dict[intent]
        available_script = self.script[intent]
        wording = random_pick(list(current_grammar_p_dict.keys()), list(current_grammar_p_dict.values()))

        # get basic scene
        scene_name = ' '.join([intent, str(self.desired_entity), wording])
        scene_content = list()

        for turn in available_script[wording]:
            if type(turn) is list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_content = [turn.replace('entity', str(self.desired_entity)) for turn in scene_content]

        if intent == 'discountURL':
            if wording == 'lack_entity' and self.given_entity is True:
                scene_content = [scene_content[0], scene_content[-1]]
            else:
                self.given_entity = True

        scene = {scene_name: scene_content}
        return scene

    def episode_generator(self, desired_entity):
        self.init_episode(desired_entity)

        # make topic coherent
        scene_dict = {'pay': [], 'express': []}
        for intent in self.picked_intents:
            if intent in ['discountURL', 'payment']:
                scene_dict['pay'].append(intent)
            else:
                scene_dict['express'].append(intent)
        scene_keys = list(scene_dict.keys())
        random.shuffle(scene_keys)

        picked_intents = list()
        for key in scene_keys:
            scene_content = scene_dict[key]
            random.shuffle(scene_content)
            picked_intents.extend(scene_content)

        # generate episode
        for intent in picked_intents:
            scene = self.scene_generator(intent)
            self.episode_script.append(scene)

        return self.episode_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['in_sales']
    available_intent = AVAILABLE_INTENT_3['in_sales']
    grammar_p_dict = GRAMMAR_P_DICT['in_sales']
    intent_p_dict = INTENT_P_DICT['in_sales']
    desired_entity = 'entityId=10'

    in_sales = InSales(script, available_intent, intent_p_dict, grammar_p_dict)

    random.seed(0)
    in_sales_script = in_sales.episode_generator(desired_entity)
    for line in in_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')

    print('.......................\n')

    random.seed(1)
    in_sales_script = in_sales.episode_generator(desired_entity)
    for line in in_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')

    print('.......................\n')

    random.seed(2)
    in_sales_script = in_sales.episode_generator(desired_entity)
    for line in in_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
