# coding=utf-8


"""
Generate reasonable episodes in customer service.

There are five principles:
1. All episodes are based on basic scenes defined in scrip.txt
2. The context is important. Some turns can be resolved according to context.
3. Defining reasonable episodes is non-trivial. We must verify the rationalisation.
4. Dialog flows based on entity, attribute, intent.
5. Not all basic scenes are available. We hold a switch to control that.

We think such episodes are reasonable:
    pre_sales, in_sales, after_sales, pre_sales + in_sales, sentiment + all combinations aforementioned
"""

from pre_sales import PreSales
from in_sales import InSales
from after_sales import AfterSales
from sentiment import Sentiment
import random
import json
from config import *
import os


class EpisodeGenerator(object):
    def __init__(self, script_file, available_intent, intent_p_dict, grammar_p_dict):
        # Init basic episode generator
        with open(os.path.join(DATA_ROOT, script_file), 'rb') as f:
            script = json.load(f)

        self.pre_sales = PreSales(script['pre_sales'],
                                  available_intent['pre_sales'],
                                  intent_p_dict['pre_sales'],
                                  grammar_p_dict['pre_sales'])
        self.in_sales = InSales(script['in_sales'],
                                available_intent['in_sales'],
                                intent_p_dict['in_sales'],
                                grammar_p_dict['in_sales'])
        self.after_sales = AfterSales(script['after_sales'],
                                      available_intent['after_sales'],
                                      intent_p_dict['after_sales'])
        self.sentiment = Sentiment(script['sentiment'])

    def sample_user(self):
        """
        Sample 2~3 candidate entities and 2~3 user concern attributes.
        """
        price = random.sample


    def episode_generator(self):
