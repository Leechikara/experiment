# coding=utf-8
import random, os, json
from utils import random_pick, filter_p_dict
from config import *


def in_sale_generator(script, intent, grammar_p_dict, given_entity, entity):
    wording = random_pick(list(grammar_p_dict.keys()), list(grammar_p_dict.values()))

    # get basic scene
    scene_name = ' '.join([intent, str(entity), wording])
    scene_content = list()

    for turn in script[wording]:
        if type(turn) is list:
            scene_content.append(random.choice(turn))
        else:
            scene_content.append(turn)

    scene_content = [turn.replace('entity', str(entity)) for turn in scene_content]

    if intent == 'discountURL':
        if wording == 'lack_entity' and given_entity is True:
            scene_content = [scene_content[0], scene_content[-1]]
        else:
            given_entity = True

    scene = {scene_name: scene_content}
    return scene, given_entity


def in_sales_controller(script, available_intent, grammar_p_dict, intent_p_dict, desired_entity):
    available_intent_p_dict = filter_p_dict(available_intent, intent_p_dict)
    pick_num = random.randint(1, len(available_intent))
    picked_intents = random_pick(list(available_intent_p_dict.keys()), list(available_intent_p_dict.values()), pick_num)
    if type(picked_intents) is not list:
        picked_intents = [picked_intents]

    # make topic coherent
    scene_dict = {'pay': [], 'express': []}
    for intent in picked_intents:
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

    # generate scene
    given_entity = False
    in_sales_script = list()

    for intent in picked_intents:
        scene, given_entity = in_sale_generator(script[intent], intent, grammar_p_dict[intent], given_entity,
                                                desired_entity)
        in_sales_script.append(scene)

    return in_sales_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, 'script.txt')
    with open(script_f, 'rb') as f:
        script = json.load(f)
    script = script['scenes']['in_sales']
    available_intent = AVAILABLE_INTENT_3['in_sales']
    grammar_p_dict = GRAMMAR_P_DICT['in_sales']
    intent_p_dict = INTENT_P_DICT['in_sales']
    desired_entity = 'entityId=10'

    in_sales_script = in_sales_controller(script, available_intent, grammar_p_dict, intent_p_dict, desired_entity)
    for line in in_sales_script:
        for l in list(line.values())[0]:
            print(l)
        print('')
