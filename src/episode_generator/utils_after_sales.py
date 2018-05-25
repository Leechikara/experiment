# coding=utf-8
import random, copy, os, json
from utils import filter_p_dict, random_pick
from config import *


def sample_scene(script):
    script_copy = copy.deepcopy(script)
    scene_name = list()
    scene_content = list()

    while type(script_copy) == dict:
        keys = list(script_copy.keys())
        random_key = random.choice(keys)
        scene_name.append(random_key)
        script_copy = script_copy[random_key]

    for turn in script_copy:
        if type(turn) == list:
            scene_content.append(random.choice(turn))
        else:
            scene_content.append(turn)

    scene_name = ' '.join(scene_name)
    scene = {scene_name: scene_content}
    return scene


def after_sales_controller(script, available_intent, intent_p_dict):
    available_intent_p_dict = filter_p_dict(available_intent, intent_p_dict)
    intents = random_pick(list(available_intent_p_dict.keys()), list(available_intent_p_dict.values()))
    intents = 'exchange_exchange'
    intent_list = intents.split('_')
    after_sales_script = list()

    if len(intent_list) == 1:
        intent = intent_list[0]
        scene = sample_scene(script[intent])
        after_sales_script.append(scene)
    else:
        intent1 = intent_list[0]
        intent2 = intent_list[1]
        scene1 = sample_scene(script[intent1])
        if intent1 == 'consult' and intent2 == 'refund':
            scene2 = sample_scene(script[intent2]['withConsult'])
        else:
            scene2 = sample_scene(script[intent2]['color']['withExchange'])
        after_sales_script.append(scene1)
        after_sales_script.append(scene2)

    return after_sales_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['scenes']['after_sales']
    available_intent = AVAILABLE_INTENT_4['after_sales']
    intent_p_dict = INTENT_P_DICT['after_sales']

    after_sales_script = after_sales_controller(script, available_intent, intent_p_dict)

    for line in after_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
