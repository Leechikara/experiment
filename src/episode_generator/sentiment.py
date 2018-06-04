# coding = utf-8
import random
import copy
import sys
import re
from config import *
from src.knowledge_base.knowledge_base import KnowledgeBase
from collections import OrderedDict, defaultdict
from utils import find_attr_entity


class Sentiment(object):
    def __init__(self, script):
        self.script = self._flatten(script)
        self.kb_helper = KnowledgeBase()

    # Roll out the content of a dictionary
    def _flatten(self, dictionary, prefix=""):
        flatten_dict = dict()
        for key, value in dictionary.items():
            if type(value) != dict:
                flatten_dict[" ".join([prefix, str(key)]).strip()] = value
            else:
                flatten_dict.update(self._flatten(value, " ".join([prefix, str(key)]).strip()))
        return flatten_dict

    def candidate_scene_generator(self, scene, last_scene):
        """
        Find all possible sentiment scene for current basic scene.
        scene is the name of basic script.
        """
        scene_element = scene.split(" ")
        scene_element = [ele.replace("attr=", "") for ele in scene_element]

        # Get all keys for sentiment scene
        sentiment_element = list()

        # A special case for refund, explict declare the attr!
        if "refund" in scene_element and "withConsult" in scene_element and last_scene is not None:
            last_scene_element = last_scene.split(" ")
            attr = None
            for ele in last_scene_element:
                if ele in SENTIMENT_TARGET:
                    attr = ele
                    break
            sentiment_element.append(attr)

        if "qa" in scene_element or "confirm" in scene_element or "compare" in scene_element:
            sentiment_element.append("pre_sales")
        elif "refund" in scene_element:
            sentiment_element.append("refund")
        elif "consult" in scene_element:
            sentiment_element.append("consult")
        elif "exchange" in scene_element:
            sentiment_element.append("exchange")
        for target in SENTIMENT_TARGET:
            if target in scene_element:
                sentiment_element.append(target)

        # This scene do not have any sentiment!
        if len(sentiment_element) == 0:
            return None

        # Get all script meet the keys
        sentiment_element = set(sentiment_element)
        sentiment_scene = dict()
        for key, value in self.script.items():
            if sentiment_element <= set(key.split(" ")):
                sentiment_scene[key] = copy.deepcopy(value)

        # No pre-defined script, for example os in pre_sales
        if len(sentiment_scene) == 0:
            return None
        return sentiment_scene

    def episode_generator(self, episode_script, episode, **kwargs):
        """
        We create a sentiment script based on generated episode script from basic 4 scene.
        """
        scene_list = list(episode_script.keys())
        candidate_sentiment = OrderedDict.fromkeys(scene_list)
        last_scene = None
        for scene in scene_list:
            sentiment_scene = self.candidate_scene_generator(scene, last_scene)
            candidate_sentiment[scene] = sentiment_scene
            last_scene = scene

        ###############################################################
        # Now, we add rules to keep the meaning sentiment scene.
        ###############################################################

        # We need delete some sentiment.
        if episode == "pre_sales":
            max_sentiment = len(kwargs["sample_entity"]) * len(kwargs["sample_goods_attr"]) / 2
        elif episode == "in_sales":
            max_sentiment = 2
        elif episode == "after_sales":
            max_sentiment = 1
        elif episode == "pre_sales in_sales":
            max_sentiment = len(kwargs["sample_entity"]) * len(kwargs["sample_goods_attr"]) / 2 + 1
        else:
            sys.exit("Undefined episode!")
        sentiment_num = random.randint(1, max_sentiment)
        available_scene = list()
        for key, value in candidate_sentiment.items():
            if value is not None:
                available_scene.append(key)
        sampled_scene = random.sample(available_scene,
                                      sentiment_num if len(available_scene) > sentiment_num else len(available_scene))
        for key in candidate_sentiment.keys():
            if key not in sampled_scene:
                candidate_sentiment[key] = None

        # Make sentiment coherent with content. Keep only a sentiment script from the candidate.
        polarity_list = list()
        for key, value in candidate_sentiment.items():
            if value is None:
                continue

            entity = None
            if len(value) > 1:
                scene_element = key.split(" ")
                if "qa" in scene_element or "confirm" in scene_element \
                        or "compare" in scene_element or "discountURL" in scene_element:
                    # find current entity and attr
                    entity, attr = find_attr_entity(scene_element)
                    if type(entity) == list:
                        entity = random.choice(entity)

                    # find a true polarity for current attr and entity
                    if attr.find("discount") == -1:
                        # for attr not discountURL and discountValue
                        if type(kwargs["attr_entity_table"][attr]) != list \
                                and kwargs["attr_entity_table"][attr] == entity \
                                or type(kwargs["attr_entity_table"][attr]) == list \
                                        and entity in kwargs["attr_entity_table"][attr]:
                            candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("positive") != -1}
                            polarity_list.append("positive")
                        else:
                            candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("negative") != -1}
                            polarity_list.append("negative")
                    else:
                        # discountURL and discountValue is special
                        assert len(candidate_sentiment[key]) != 1
                        kb_results = self.kb_helper.inform("discountValue", [entity])
                        if kb_results["discountValue"][entity] is not None:
                            if "attr_entity_table" in kwargs.keys() and \
                                            "discountValue" in kwargs["attr_entity_table"].keys():
                                if kwargs["attr_entity_table"]["discountValue"] == entity:
                                    candidate_sentiment[key] = {k: v for k, v in value.items()
                                                                if k.find("positive") != -1}
                                    polarity_list.append("positive")
                                else:
                                    candidate_sentiment[key] = {k: v for k, v in value.items()
                                                                if k.find("notNone") != -1
                                                                and k.find("negative") != -1}
                                    polarity_list.append("negative")
                            else:
                                if random.randint(8, 9) > kb_results["discountValue"][entity]:
                                    candidate_sentiment[key] = {k: v for k, v in value.items()
                                                                if k.find("positive") != -1}
                                    polarity_list.append("positive")
                                else:
                                    candidate_sentiment[key] = {k: v for k, v in value.items()
                                                                if k.find("notNone") != -1
                                                                and k.find("negative") != -1}
                                    polarity_list.append("negative")
                        else:
                            candidate_sentiment[key] = {k: v for k, v in value.items()
                                                        if k.find("notNone") == -1
                                                        and k.find("negative") != -1}
                            polarity_list.append("negative")
                elif "expressTime" in scene_element or "expressInfo" in scene_element:
                    # random give a sentiment polarity
                    if random.random() < 0.5:
                        candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("positive") != -1}
                        polarity_list.append("positive")
                    else:
                        candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("negative") != -1}
                        polarity_list.append("negative")
                elif "pre_sales_end" in scene_element:
                    # make sentiment coherent
                    if len(set(polarity_list)) == 2:
                        candidate_sentiment[key] = None
                        continue
                    elif len(set(polarity_list)) == 0 and random.random() < 0.5 or set(polarity_list) == {"positive"}:
                        candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("positive") != -1}
                        polarity_list.append("positive")
                    else:
                        candidate_sentiment[key] = {k: v for k, v in value.items() if k.find("negative") != -1}
                        polarity_list.append("negative")
                elif "refund" in scene_element:
                    # random choice one
                    random_reason = random.choice(list(value.items()))
                    candidate_sentiment[key] = {random_reason[0]: random_reason[1]}
                    polarity_list.append("negative")
                else:
                    sys.exit("Unconsidered situations happen!")

            assert len(candidate_sentiment[key]) == 1
            scene_content = list()
            scene_name = list(candidate_sentiment[key].items())[0][0]
            for turn in list(candidate_sentiment[key].items())[0][1]:
                if type(turn) == list:
                    scene_content.append(random.choice(turn))
                else:
                    scene_content.append(turn)

            # Replace entity. Note in some case the entity can not be replace!
            if entity is not None:
                scene_content = [re.sub(r"\$entity\$", "$entityId=" + str(entity) + "$", turn)
                                 for turn in scene_content]
            candidate_sentiment[key][scene_name] = scene_content

        # keep only sentiment scene for the same attr and polarity
        sentiment_dict = defaultdict(list)
        for key, value in candidate_sentiment.items():
            if candidate_sentiment[key] is None:
                continue
            else:
                sentiment_dict[list(value.keys())[0]].append(key)
        for key, value in sentiment_dict.items():
            if len(value) > 1:
                keep_scene = random.choice(value)
                for scene in value:
                    if scene != keep_scene:
                        candidate_sentiment[scene] = None

        episode_script = self.organize_script(episode_script, candidate_sentiment)
        return episode_script

    @staticmethod
    def organize_script(episode_script, candidate_sentiment):
        sentiment_script = OrderedDict()
        for scene_name in episode_script.keys():
            if candidate_sentiment[scene_name] is None:
                sentiment_script[scene_name] = episode_script[scene_name]
            else:
                sentiment_scene_name = list(candidate_sentiment[scene_name].keys())[0]
                sentiment_scene_content = list(candidate_sentiment[scene_name].values())[0]
                # what's the polarity
                if sentiment_scene_name.find("positive") != -1:
                    polarity = "positive"
                else:
                    polarity = "negative"

                new_scene_name = " ".join([polarity, scene_name])
                if episode_script[scene_name] is None:
                    # replace rule
                    sentiment_script[new_scene_name] = sentiment_scene_content
                else:
                    if sentiment_scene_name.find("pre_sales") != -1 or sentiment_scene_name.find("express") != -1 \
                            or sentiment_scene_name.find("URL") != -1:
                        # append rule
                        episode_script[scene_name].extend(sentiment_scene_content)
                        sentiment_script[new_scene_name] = episode_script[scene_name]
                    elif sentiment_scene_name.find("refund") != -1 or sentiment_scene_name.find("consult") != -1:
                        # choice a rule from feasible rule set
                        sentiment_rules = copy.deepcopy(SENTIMENT_RULES)
                        if scene_name.find("refund") != -1:
                            sentiment_rules.remove("append")
                        rule = random.choice(sentiment_rules)
                        if rule == "append":
                            episode_script[scene_name].extend(sentiment_scene_content)
                            sentiment_script[new_scene_name] = episode_script[scene_name]
                        elif rule == "prefix":
                            sentiment_scene_content.extend(episode_script[scene_name])
                            sentiment_script[new_scene_name] = sentiment_scene_content
                            # user have given the refund reason
                            if sentiment_scene_name.find("refund") != -1 and scene_name.find("complete") != -1:
                                del sentiment_script[new_scene_name][3:5]
                        elif rule == "insert":
                            # insert must happen after agent turn
                            if sentiment_scene_name.find("refund") != -1:
                                if scene_name.find("withConsult") == -1:
                                    # in this situation, we can only replace $refundReason$
                                    episode_script[scene_name].remove("$refundReason$")
                                    episode_script[scene_name].remove("麻烦您提供一下您的订单号")
                                    episode_script[scene_name].insert(2, sentiment_scene_content[0])
                                    episode_script[scene_name].insert(3, " ".join([sentiment_scene_content[1],
                                                                                   "麻烦您提供一下您的订单号"]))
                                    sentiment_script[new_scene_name] = episode_script[scene_name]
                                else:
                                    if random.random() < 0.5:
                                        episode_script[scene_name].extend(sentiment_scene_content)
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                    else:
                                        sentiment_scene_content.extend(episode_script[scene_name])
                                        sentiment_script[new_scene_name] = sentiment_scene_content
                            else:
                                if scene_name.find("verbose1") != -1:
                                    if random.random() < 0.5:
                                        episode_script[scene_name].extend(sentiment_scene_content)
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                    else:
                                        sentiment_scene_content.extend(episode_script[scene_name])
                                        sentiment_script[new_scene_name] = sentiment_scene_content
                                elif scene_name.find("verbose2") != -1:
                                    p = random.random()
                                    if p < 1 / 3:
                                        episode_script[scene_name].extend(sentiment_scene_content)
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                    elif p < 2 / 3:
                                        sentiment_scene_content.extend(episode_script[scene_name])
                                        sentiment_script[new_scene_name] = sentiment_scene_content
                                    else:
                                        episode_script[scene_name].insert(2, sentiment_scene_content[0])
                                        episode_script[scene_name].insert(3, sentiment_scene_content[1])
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                else:
                                    if len(episode_script[scene_name]) == 4:
                                        p = random.random()
                                        if p < 1 / 3:
                                            episode_script[scene_name].extend(sentiment_scene_content)
                                            sentiment_script[new_scene_name] = episode_script[scene_name]
                                        elif p < 2 / 3:
                                            sentiment_scene_content.extend(episode_script[scene_name])
                                            sentiment_script[new_scene_name] = sentiment_scene_content
                                        else:
                                            episode_script[scene_name].insert(2, sentiment_scene_content[0])
                                            episode_script[scene_name].insert(3, sentiment_scene_content[1])
                                            sentiment_script[new_scene_name] = episode_script[scene_name]
                                    else:
                                        if random.random() < 0.5:
                                            episode_script[scene_name].insert(2, sentiment_scene_content[0])
                                            episode_script[scene_name].insert(3, sentiment_scene_content[1])
                                            sentiment_script[new_scene_name] = episode_script[scene_name]
                                        else:
                                            episode_script[scene_name].insert(4, sentiment_scene_content[0])
                                            del episode_script[scene_name][5:7]
                                            episode_script[scene_name].insert(5, " ".join([sentiment_scene_content[1],
                                                                                           "麻烦您提供一下姓名"]))
                                            sentiment_script[new_scene_name] = episode_script[scene_name]
                        else:
                            if sentiment_scene_name.find("refund") != -1:
                                temp_list = list()
                                temp_list.append(sentiment_scene_content[0])
                                temp_list.append(episode_script[scene_name][0])
                                if new_scene_name.find("withConsult") == -1:
                                    del episode_script[scene_name][0:4]
                                else:
                                    del episode_script[scene_name][0:2]
                                episode_script[scene_name].insert(0, " ".join(temp_list))
                                episode_script[scene_name].insert(1, " ".join([sentiment_scene_content[1],
                                                                               "麻烦您提供一下您的订单号"]))
                                sentiment_script[new_scene_name] = episode_script[scene_name]
                            else:
                                if scene_name.find("verbose1") != -1:
                                    temp_list = list()
                                    temp_list.append(sentiment_scene_content[0])
                                    temp_list.append(episode_script[scene_name][0])
                                    del episode_script[scene_name][0:2]
                                    episode_script[scene_name].insert(0, " ".join(temp_list))
                                    episode_script[scene_name].insert(1, " ".join([sentiment_scene_content[1],
                                                                                   "您可以尝试更新到最新的系统"]))
                                    sentiment_script[new_scene_name] = episode_script[scene_name]
                                elif scene_name.find("verbose2") != -1:
                                    if random.random() < 0.5:
                                        temp_list = list()
                                        temp_list.append(sentiment_scene_content[0])
                                        temp_list.append(episode_script[scene_name][0])
                                        del episode_script[scene_name][0:2]
                                        episode_script[scene_name].insert(0, " ".join(temp_list))
                                        episode_script[scene_name].insert(1, " ".join([sentiment_scene_content[1],
                                                                                       "您可以尝试更新到最新的系统"]))
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                    else:
                                        temp_list = list()
                                        temp_list.append(sentiment_scene_content[0])
                                        temp_list.append(episode_script[scene_name][2])
                                        del episode_script[scene_name][2:4]
                                        episode_script[scene_name].insert(2, " ".join(temp_list))
                                        episode_script[scene_name].insert(3, " ".join([sentiment_scene_content[1],
                                                                                       "$osUpdate$"]))
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                else:
                                    if random.random() < 0.5:
                                        temp_list = list()
                                        temp_list.append(sentiment_scene_content[0])
                                        temp_list.append(episode_script[scene_name][0])
                                        del episode_script[scene_name][0:2]
                                        episode_script[scene_name].insert(0, " ".join(temp_list))
                                        episode_script[scene_name].insert(1, " ".join([sentiment_scene_content[1],
                                                                                       "您可以尝试更新到最新的系统"]))
                                        sentiment_script[new_scene_name] = episode_script[scene_name]
                                    else:
                                        temp_list = list()
                                        temp_list.append(sentiment_scene_content[0])
                                        temp_list.append(episode_script[scene_name][2])
                                        del episode_script[scene_name][2:4]
                                        episode_script[scene_name].insert(2, " ".join(temp_list))
                                        episode_script[scene_name].insert(3, " ".join([sentiment_scene_content[1],
                                                                                       "我帮您返厂维修一下行吗"]))
                                        sentiment_script[new_scene_name] = episode_script[scene_name]

                    else:
                        sys.exit("Unconsidered situations happened!")
        return sentiment_script
