# coding=utf-8


"""
Generate reasonable episodes in customer service.

There are five principles:
1. All episodes are based on basic scenes defined in scrip.txt
2. The context is important. Some turns can be resolved according to context.
3. Defining reasonable episodes is non-trivial. We must verify the rationalisation.
4. Dialog flows based on entity, attribute, intent.
5. In our experiment, not all basic scenes are available. We hold a switch to control that.

We think such episodes are reasonable:
    pre_sales, in_sales, after_sales, pre_sales + in_sales, sentiment + all combinations aforementioned
"""

from pre_sales import PreSales
from in_sales import InSales
from after_sales import AfterSales
from sentiment import Sentiment
from src.knowledge_base.knowledge_base import KnowledgeBase
from src.translator.translator import Translator
from config import *
from utils import find_attr_entity
from collections import OrderedDict
import random
import json
import copy
import os
import sys
import io


class EpisodeGenerator(object):
    def __init__(self, available_intent,
                 script_file="script.txt",
                 intent_p_dict=INTENT_P_DICT,
                 grammar_p_dict=GRAMMAR_P_DICT):
        self.episode_script = None

        # Init basic episode generator
        with open(os.path.join(DATA_ROOT, script_file), "rb") as f:
            script = json.load(f)

        if "pre_sales" in available_intent.keys():
            self.pre_sales = PreSales(script["pre_sales"],
                                      available_intent["pre_sales"],
                                      intent_p_dict["pre_sales"],
                                      grammar_p_dict["pre_sales"])
        else:
            self.pre_sales = None
        if "in_sales" in available_intent.keys():
            self.in_sales = InSales(script["in_sales"],
                                    available_intent["in_sales"],
                                    intent_p_dict["in_sales"],
                                    grammar_p_dict["in_sales"])
        else:
            self.in_sales = None
        if "after_sales" in available_intent.keys():
            self.after_sales = AfterSales(script["after_sales"],
                                          available_intent["after_sales"],
                                          intent_p_dict["after_sales"])
        else:
            self.after_sales = None
        if "sentiment" in available_intent.keys():
            self.sentiment = Sentiment(script["sentiment"])
        else:
            self.sentiment = None

        # Get available episodes
        self.available_episode = list(available_intent.keys())
        if "pre_sales" in self.available_episode and "in_sales" in self.available_episode:
            self.available_episode.append(" ".join(["pre_sales", "in_sales"]))

        # Knowledge Base
        self.kb_helper = KnowledgeBase()

        # translate agent dialog action according to KB results
        self.translator = Translator()

    def init_episode(self):
        self.episode_script = None

    def sample_user(self):
        """
        Sample 2~3 candidate entities and 2~3 user concern attributes.
        We use this user model to get pre_sales and in_sales scripts.
        """
        # get the candidate entities according to a price range
        candidate_entity = list()
        while len(candidate_entity) < 3:
            price = random.randint(100, 200) * 10
            upper = price + PRICE_NOISE
            lower = price - PRICE_NOISE
            candidate_entity = self.kb_helper.search_kb(attr="price", dtype="int", compare="~", upper=upper,
                                                        lower=lower)

        # Sample entities and attr. We restrict it in a range!
        entity_num = random.choice([2, 3])
        attr_num = random.choice([2, 3])
        if entity_num == 3 and attr_num == 3:
            if random.random() < 0.5:
                entity_num -= 1
            else:
                attr_num -= 1
        sample_entity = random.sample(candidate_entity, min(entity_num, len(candidate_entity)))
        sample_goods_attr = random.sample(PRE_SALES_ATTR, min(attr_num, len(PRE_SALES_ATTR)))

        # Get the priority of attr
        attr_priority = copy.deepcopy(sample_goods_attr)
        random.shuffle(attr_priority)

        # Define incommensurable constrains: color and os
        hard_constrains = dict()
        if "color" in sample_goods_attr:
            color_list = list()
            for entity in sample_entity:
                color_list.extend(entity["color"])
            color_list = list(set(color_list))
            hard_constrains["color"] = random.choice(color_list)
        if "os" in sample_goods_attr:
            os_list = list()
            for entity in sample_entity:
                os_list.append(entity["os"])
            os_list = list(set(os_list))
            hard_constrains["os"] = random.choice(os_list)

        sample_entity = ["entityId=" + str(entity["id"]) for entity in sample_entity]

        return sample_entity, sample_goods_attr, attr_priority, hard_constrains

    def sample_desired_entity(self):
        entity = random.choice(self.kb_helper.kb)
        entity_id = "entityId=" + str(entity["id"])
        return entity_id

    def calculate_desired_entity(self, sample_entity, sample_goods_attr, attr_priority, hard_constrains):
        sample_entity = [self.kb_helper.find_entity(entity) for entity in sample_entity]

        # For each attr, which entities meet the constrain or which entity is the best one
        attr_entity_table = dict.fromkeys(sample_goods_attr, None)

        # Take all attr into consideration.
        for attr in sample_goods_attr:
            # Record attr_value of each entity
            entity_attr_value = dict()
            for entity in sample_entity:
                entity_attr_value[entity["id"]] = entity[attr]

            # Now we judge which entity is better
            if "prefer" in GOODS_ATTRIBUTE_DEFINITION[attr].keys():
                # What kind of value do user prefer
                if GOODS_ATTRIBUTE_DEFINITION[attr]["prefer"] == "low":
                    reverse = False
                else:
                    reverse = True

                if GOODS_ATTRIBUTE_DEFINITION[attr]["dtype"] in ["int", "float"]:
                    entity_id = sorted(entity_attr_value.items(),
                                       key=lambda item: item[1] if item[1] is not None else 1,
                                       reverse=reverse)[0][0]
                elif GOODS_ATTRIBUTE_DEFINITION[attr]["dtype"] in ["str", "bool"]:
                    value_range = GOODS_ATTRIBUTE_DEFINITION[attr]["range"]
                    entity_id = sorted(entity_attr_value.items(),
                                       key=lambda item: value_range.index(item[1]),
                                       reverse=reverse)[0][0]
                else:
                    sys.exit("Unconsidered situations happen!")
                attr_entity_table[attr] = entity_id
            else:
                # We check incommensurable constrains
                assert attr in hard_constrains.keys()
                attr_entity_table[attr] = list()
                for entity_id, value in entity_attr_value.items():
                    if type(value) == list and hard_constrains[attr] in value:
                        attr_entity_table[attr].append(entity_id)
                    elif type(value) == str and hard_constrains[attr] == value:
                        attr_entity_table[attr].append(entity_id)

        # Now, we calculate the best entity according to attr_priority
        entity_score = dict()
        for entity in sample_entity:
            score = 0
            for attr, entity_id in attr_entity_table.items():
                if type(entity_id) == list:
                    if entity["id"] in entity_id:
                        score += 1 + 0.15 * attr_priority.index(attr)
                else:
                    if entity["id"] == entity_id:
                        score += 1 + 0.15 * attr_priority.index(attr)
            entity_score[entity["id"]] = score

        entity_id = "entityId=" + str(sorted(entity_score.items(), key=lambda item: item[1], reverse=True)[0][0])
        return entity_id, attr_entity_table

    def episode_generator(self):
        self.init_episode()

        # First, we decide which dialog episode to generate
        episode = random.choice(self.available_episode)

        # Next, we decide if sentiment is available
        if episode == "sentiment":
            decorate_sentiment = True
            self.available_episode.remove("sentiment")
            episode = random.choice(self.available_episode)
            self.available_episode.append("sentiment")
        else:
            decorate_sentiment = False

        # Then, generate basic script
        if episode == "pre_sales":
            sample_entity, sample_goods_attr, attr_priority, hard_constrains = self.sample_user()
            episode_script = self.pre_sales.episode_generator(sample_goods_attr, sample_entity)
            attr_entity_table = None
        elif episode == "in_sales":
            desired_entity = self.sample_desired_entity()
            episode_script = self.in_sales.episode_generator(desired_entity)
        elif episode == "after_sales":
            episode_script = self.after_sales.episode_generator()
        elif episode == "pre_sales in_sales":
            sample_entity, sample_goods_attr, attr_priority, hard_constrains = self.sample_user()
            episode_script = self.pre_sales.episode_generator(sample_goods_attr, sample_entity)
            desired_entity, attr_entity_table = self.calculate_desired_entity(sample_entity, sample_goods_attr,
                                                                              attr_priority, hard_constrains)
            episode_script.update(self.in_sales.episode_generator(desired_entity))
        else:
            sys.exit("Unconsidered situations happen!")

        # And then, we add sentiment factor.
        if decorate_sentiment:
            if episode.find("pre_sales") != -1:
                if attr_entity_table is None:
                    _, attr_entity_table = self.calculate_desired_entity(sample_entity, sample_goods_attr,
                                                                         attr_priority, hard_constrains)
                episode_script = self.sentiment.episode_generator(episode_script, episode,
                                                                  sample_entity=sample_entity,
                                                                  sample_goods_attr=sample_goods_attr,
                                                                  attr_entity_table=attr_entity_table)
            else:
                episode_script = self.sentiment.episode_generator(episode_script, episode)

        # In the end, we Instantiate all free content. We use this script to train our model!
        self.episode_script = self.instantiation(episode_script)

        return self.episode_script

    def instantiation(self, episode_script):
        # instantiation all unknown content
        # special for unconstrained entity, compare operation and compare value
        candidate_entity = None
        for key, value in episode_script.items():
            if key.find("confirm") != -1:
                for i, turn in enumerate(value):
                    if turn.find("$value$") != -1:
                        _, attr = find_attr_entity(key)
                        assert attr in CONFIRM_PERMITTED_ATTR
                        if "range" in GOODS_ATTRIBUTE_DEFINITION[attr].keys():
                            candidate_value = random.choice(GOODS_ATTRIBUTE_DEFINITION[attr]["range"])
                        elif "min" in GOODS_ATTRIBUTE_DEFINITION[attr].keys() and \
                                        "max" in GOODS_ATTRIBUTE_DEFINITION[attr].keys():
                            candidate_value = random.choice(list(range(GOODS_ATTRIBUTE_DEFINITION[attr]["min"],
                                                                       GOODS_ATTRIBUTE_DEFINITION[attr]["max"])))
                        else:
                            sys.exit("Unconsidered situations happen!")
                        if type(candidate_value) == int:
                            if candidate_value < 10:
                                pass
                            elif candidate_value < 100:
                                candidate_value = candidate_value // 5 * 5
                            else:
                                candidate_value = candidate_value // 100 * 100
                        elif type(candidate_value) == float:
                            candidate_value = candidate_value * 10 // 5 * 5 * 0.1
                        else:
                            pass
                        if "unit" in GOODS_ATTRIBUTE_DEFINITION[attr].keys():
                            candidate_value = str(candidate_value) + GOODS_ATTRIBUTE_DEFINITION[attr]["unit"]
                        else:
                            candidate_value = str(candidate_value)
                        episode_script[key][i] = episode_script[key][i].replace("$value$", candidate_value)
                    if turn.find("$compare$") != -1:
                        candidate_compare = random.choice(["是", "不低于", "不少于", "低于", "少于",
                                                           "小于", "没到", "不到", "高于", "大于", "多于"])
                        episode_script[key][i] = episode_script[key][i].replace("$compare$", candidate_compare)
            else:
                for i, turn in enumerate(value):
                    if turn.find("$entity$") != -1:
                        candidate_entity = random.choice(self.kb_helper.kb)
                        episode_script[key][i] = episode_script[key][i].replace("$entity$",
                                                                                "entityId=" + str(
                                                                                    candidate_entity["id"]))
                    if turn.find("$color$") != -1:
                        if candidate_entity is not None:
                            candidate_color = random.choice(candidate_entity["color"])
                        else:
                            candidate_color = random.choice(GOODS_ATTRIBUTE_DEFINITION["color"]["range"])
                        episode_script[key][i] = episode_script[key][i].replace("$color$", candidate_color)
        return episode_script

    def show(self):
        # Print the episode script to screen.
        episode_script = OrderedDict()
        for key, content in self.episode_script.items():
            temp_content = list()
            for i, turn in enumerate(content):
                if i % 2 == 1:
                    temp_content.append(self.translator.translate(turn))
                else:
                    temp_content.append(turn)
            episode_script[key] = temp_content
        episode_script = json.dumps(episode_script, ensure_ascii=False, indent=2)
        print(episode_script)


if __name__ == "__main__":
    episode_generator = EpisodeGenerator(AVAILABLE_INTENT_3)

    # test our code
    random.seed(1)
    with io.open(os.path.join(DATA_ROOT, "log.txt"), "w", encoding="utf-8") as f:
        for _ in range(10000):
            f.write(str(_))
            f.write(".......................\n")
            episode_script = episode_generator.episode_generator()
            for line in episode_script.values():
                for l in line:
                    f.write(l + "\n")
