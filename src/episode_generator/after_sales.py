# coding=utf-8
import random, copy, os, json
from utils import filter_p_dict, random_pick
from config import *


class AfterSales(object):
    def __init__(self, script, available_intent, intent_p_dict):
        self.script = script
        self.available_intent = available_intent
        self.intent_p_dict = intent_p_dict
        self.available_intent_p_dict = filter_p_dict(self.available_intent, self.intent_p_dict)

    def scene_generator(self, *args):
        available_script = copy.deepcopy(self.script)
        for arg in args:
            available_script = available_script[arg]
        with_permitted = False if len(args) == 1 else True

        scene_name = list()
        scene_content = list()

        while type(available_script) == dict:
            keys = list(available_script.keys())
            if with_permitted is False:
                if 'withConsult' in keys:
                    keys.remove('withConsult')
                if 'withExchange' in keys:
                    keys.remove('withExchange')
            random_key = random.choice(keys)
            scene_name.append(random_key)
            available_script = available_script[random_key]

        for turn in available_script:
            if type(turn) == list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_name = ' '.join(scene_name)
        scene = {scene_name: scene_content}
        return scene

    def episode_generator(self):
        intent = random_pick(list(self.available_intent_p_dict.keys()),
                             list(self.available_intent_p_dict.values()))
        intent_list = intent.split('_')
        episode_script = list()

        if len(intent_list) == 1:
            intent = intent_list[0]
            scene = self.scene_generator(intent)
            episode_script.append(scene)
        else:
            intent1 = intent_list[0]
            intent2 = intent_list[1]
            scene1 = self.scene_generator(intent1)
            if intent1 == 'consult' and intent2 == 'refund':
                scene2 = self.scene_generator(intent2, 'withConsult')
            else:
                scene2 = self.scene_generator(intent2, 'color', 'withExchange')
            episode_script.append(scene1)
            episode_script.append(scene2)

            # In such case, we should delete exchange scene
            if intent1 == 'exchange' and intent2 == 'exchange' and list(scene1.keys())[0].find('verbose3') == -1:
                del episode_script[-1]

        return episode_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['scenes']['after_sales']
    available_intent = AVAILABLE_INTENT_5['after_sales']
    intent_p_dict = INTENT_P_DICT['after_sales']

    after_sales = AfterSales(script, available_intent, intent_p_dict)

    random.seed(0)
    after_sales_script = after_sales.episode_generator()
    for line in after_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')

    print('.......................\n')

    random.seed(1)
    after_sales_script = after_sales.episode_generator()
    for line in after_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
