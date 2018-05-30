# coding = utf-8
import random
import copy
import sys
from config import *


class Sentiment(object):
    def __init__(self, script):
        self.script = self._flatten(script)

    # Roll out the content of a dictionary
    def _flatten(self, dictionary, prefix=""):
        flatten_dict = dict()
        for key, value in dictionary.items():
            if type(value) != dict:
                flatten_dict[" ".join([prefix, str(key)]).strip()] = value
            else:
                flatten_dict.update(self._flatten(value, " ".join([prefix, str(key)]).strip()))
        return flatten_dict

    def candidate_scene_generator(self, scene):
        """
        Find all possible sentiment scene for current scene
        """
        scene_element = scene.split(" ")

        # Get all keys for sentiment scene
        sentiment_element = list()
        if "qa" in scene_element or "confirm" in scene_element or "compare" in scene_element:
            sentiment_element.append("pre_sales")
        elif "refund" in scene_element:
            sentiment_element.append("refund")
        elif "consult" in scene_element:
            sentiment_element.append("consult")
        for target in SENTIMENT_TARGET:
            if target in scene_element:
                sentiment_element.append(target)

        # This scene do not have any sentiment!
        if len(sentiment_element) == 0:
            return None

        # Get all script meet the keys
        sentiment_scene = dict()
        for key, value in self.script.items():
            match = True
            for element in sentiment_element:
                if key.find(element) == -1:
                    match = False
                    break
            if match:
                sentiment_scene[key] = copy.deepcopy(value)

        return sentiment_scene

    def decorate_sentiment(self, episode_script, episode, **kwargs):
        """
        Given a generated episode_script from basic 4 scene. We add sentiment factor into them.
        """
        scene_list = list(episode_script.keys())
        candidate_sentiment = dict.fromkeys(scene_list)
        for scene in scene_list:
            sentiment_scene = self.candidate_scene_generator(scene)
            candidate_sentiment[scene] = sentiment_scene

        ###############################################################
        # Now, we add rules to keep the meaning sentiment scene.
        # (1) We need delete some sentiment.
        # (2) User may not praise or
        # (2) Make sentiment coherent with content.
        ###############################################################

        # sample some sentiment scene
        if episode == "pre_sales":
            max_sentiment = len(kwargs['sample_entity']) * len(kwargs['sample_goods_attr']) / 2 - 1
        elif episode == "in_sales":
            max_sentiment = 2
        elif episode == "after_sales":
            max_sentiment = 2
        elif episode == "pre_sales in_sales":
            max_sentiment = len(kwargs['sample_entity']) * len(kwargs['sample_goods_attr']) / 2
        else:
            sys.exit("Undefined episode!")
        sentiment_num = random.randint(1, max_sentiment)
        available_scene = list()
        for key, value in candidate_sentiment.items():
            if value is not None:
                available_scene.append(key)
        sampled_scene = random.sample(available_scene, sentiment_num)
        for key in candidate_sentiment.keys():
            if key not in sampled_scene:
                candidate_sentiment[key] = None

        # keep only a sentiment scene that is meaningful
        for key, value in candidate_sentiment.items():
            if value is None or len(value) == 1:
                continue
            else:
                scene_element = key.split(" ")
                if "qa" in scene_element or "confirm" in scene_element:






    def reorganize_sentiment(self):
        pass
