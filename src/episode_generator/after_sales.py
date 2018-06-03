# coding=utf-8
import random, copy, os, json
from utils import filter_p_dict, random_pick
from config import *
from collections import OrderedDict


class AfterSales(object):
    def __init__(self, script, available_intent, intent_p_dict):
        self.script = script
        self.available_intent = available_intent
        self.intent_p_dict = intent_p_dict
        self.available_intent_p_dict = filter_p_dict(self.available_intent, self.intent_p_dict)
        self.episode_script = None

    def init_episode(self):
        self.episode_script = OrderedDict()

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
                if "withConsult" in keys:
                    keys.remove("withConsult")
                if "withExchange" in keys:
                    keys.remove("withExchange")
            random_key = random.choice(keys)
            scene_name.append(random_key)
            available_script = available_script[random_key]

        for turn in available_script:
            if type(turn) == list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene_name = " ".join(list(args) + scene_name)
        return scene_name, scene_content

    def episode_generator(self):
        self.init_episode()

        intent = random_pick(list(self.available_intent_p_dict.keys()),
                             list(self.available_intent_p_dict.values()))
        intent_list = intent.split("_")

        if len(intent_list) == 1:
            intent = intent_list[0]
            scene_name, scene_content = self.scene_generator(intent)
            self.episode_script[scene_name] = scene_content
        else:
            intent1 = intent_list[0]
            intent2 = intent_list[1]
            scene_name1, scene_content1 = self.scene_generator(intent1)
            if intent1 == "consult" and intent2 == "refund":
                scene_name2, scene_content2 = self.scene_generator(intent2, "withConsult")
            else:
                scene_name2, scene_content2 = self.scene_generator(intent2, "color", "withExchange")
            self.episode_script[scene_name1] = scene_content1
            self.episode_script[scene_name2] = scene_content2

            # In such case, we should delete exchange scene
            if intent1 == "exchange" and intent2 == "exchange" and scene_name1.find("verbose") != -1 \
                    and scene_name1.find("3") == -1 or scene_name1.find("color") != -1:
                key = list(self.episode_script.keys())[-1]
                del self.episode_script[key]

        return self.episode_script


if __name__ == "__main__":
    script_f = os.path.join(DATA_ROOT, "script.txt")
    with open(script_f, "rb") as f:
        script = json.load(f)
    script = script["after_sales"]
    available_intent = AVAILABLE_INTENT_5["after_sales"]
    intent_p_dict = INTENT_P_DICT["after_sales"]

    after_sales = AfterSales(script, available_intent, intent_p_dict)

    random.seed(0)
    after_sales_script = after_sales.episode_generator()
    for line in after_sales_script.values():
        for l in line:
            print(l)
        print("")

    print(".......................\n")

    random.seed(1)
    after_sales_script = after_sales.episode_generator()
    for line in after_sales_script.values():
        for l in line:
            print(l)
        print("")
