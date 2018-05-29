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
from src.knowledge_base.knowledge_base import KnowledgeBase
import random
import json
import copy
from config import *
import os
import sys


class EpisodeGenerator(object):
    def __init__(self, available_intent,
                 script_file='script.txt',
                 intent_p_dict=INTENT_P_DICT,
                 grammar_p_dict=GRAMMAR_P_DICT):
        # Init basic episode generator
        with open(os.path.join(DATA_ROOT, script_file), 'rb') as f:
            script = json.load(f)

        if 'pre_sales' in available_intent.keys():
            self.pre_sales = PreSales(script['pre_sales'],
                                      available_intent['pre_sales'],
                                      intent_p_dict['pre_sales'],
                                      grammar_p_dict['pre_sales'])
        else:
            self.pre_sales = None
        if 'in_sales' in available_intent.keys():
            self.in_sales = InSales(script['in_sales'],
                                    available_intent['in_sales'],
                                    intent_p_dict['in_sales'],
                                    grammar_p_dict['in_sales'])
        else:
            self.in_sales = None
        if 'after_sales' in available_intent.keys():
            self.after_sales = AfterSales(script['after_sales'],
                                          available_intent['after_sales'],
                                          intent_p_dict['after_sales'])
        else:
            self.after_sales = None
        if 'sentiment' in available_intent.keys():
            self.sentiment = Sentiment(script['sentiment'])
        else:
            self.sentiment = None

        # Get available episodes
        self.available_episode = list(available_intent.keys())
        if 'pre_sales' in self.available_episode and 'in_sales' in self.available_episode:
            self.available_episode.append(' '.join(['pre_sales', 'in_sales']))

        # Knowledge Base
        self.kb_helper = KnowledgeBase()

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
            candidate_entity = self.kb_helper.search_kb(attr='price', dtype='int', compare='~', upper=upper,
                                                        lower=lower)

        # sample entities
        entity_num = random.choice([2, 3])
        sample_entity = random.sample(candidate_entity, entity_num)

        # sample attr
        attr_num = random.choice([2, 3])
        sample_goods_attr = random.sample(PRE_SALES_ATTR, attr_num)

        # get the priority of attr
        attr_priority = copy.deepcopy(sample_goods_attr)
        random.shuffle(attr_priority)

        # define those hard constrains: color and os
        hard_constrains = dict()
        if 'color' in sample_goods_attr:
            color_list = list()
            for entity in sample_entity:
                color_list.extend(entity['color'])
            color_list = list(set(color_list))
            hard_constrains['color'] = random.choice(color_list)
        if 'os' in sample_goods_attr:
            os_list = list()
            for entity in sample_entity:
                os_list.append(entity['os'])
            os_list = list(set(os_list))
            hard_constrains['os'] = random.choice(os_list)

        sample_entity = ["entityId=" + str(entity['id']) for entity in sample_entity]

        return sample_entity, sample_goods_attr, attr_priority, hard_constrains

    def sample_desired_entity(self):
        entity = random.choice(self.kb_helper.kb)
        entity_id = "entityId=" + str(entity['id'])
        return entity_id

    def calculate_desired_entity(self, sample_entity, sample_goods_attr, attr_priority, hard_constrains):
        sample_entity = [self.kb_helper.find_entity(entity) for entity in sample_entity]

        # for each attr, which entities meet the constrain or which entity is the best one
        attr_entity_table = dict.fromkeys(sample_goods_attr, None)

        # take all attr into consideration.
        for attr in sample_goods_attr:
            # record attr_value of each entity
            entity_attr_value = dict()
            for entity in sample_entity:
                entity_attr_value[entity['id']] = entity[attr]

            # Now we judge which entity is better
            if 'prefer' in GOODS_ATTRIBUTE_DEFINITION[attr].keys():
                # What kind of value do user prefer
                if GOODS_ATTRIBUTE_DEFINITION[attr]['prefer'] == 'low':
                    reverse = False
                else:
                    reverse = True

                if GOODS_ATTRIBUTE_DEFINITION[attr]['dtype'] in ['int', 'float']:
                    entity_id = sorted(entity_attr_value.items(),
                                       key=lambda item: item[1] if item[1] is not None else 1,
                                       reverse=reverse)[0][0]
                elif GOODS_ATTRIBUTE_DEFINITION[attr]['dtype'] in ['str', 'bool']:
                    value_range = GOODS_ATTRIBUTE_DEFINITION[attr]['range']
                    entity_id = sorted(entity_attr_value.items(),
                                       key=lambda item: value_range.index(item[1]),
                                       reverse=reverse)[0][0]
                else:
                    sys.exit("Unconsidered situations happen!")
                attr_entity_table[attr] = entity_id
            else:
                # We check hard constrains
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
                    if entity['id'] in entity_id:
                        score += 1 + 0.15 * attr_priority.index(attr)
                else:
                    if entity['id'] == entity_id:
                        score += 1 + 0.15 * attr_priority.index(attr)
            entity_score[entity['id']] = score

        entity_id = "entityId=" + str(sorted(entity_score.items(), key=lambda item: item[1], reverse=True)[0][0])
        return entity_id

    def episode_generator(self):
        # First, we decide which dialog episode to generate
        episode = random.choice(self.available_episode)

        # Next, we decide if sentiment is available
        if episode == 'sentiment':
            decorate_sentiment = True
            self.available_episode.remove('sentiment')
            episode = random.choice(self.available_episode)
            self.available_episode.append('sentiment')
        else:
            decorate_sentiment = False

        # Then, generate basic script
        if episode == 'pre_sales':
            sample_entity, sample_goods_attr, attr_priority, hard_constrains = self.sample_user()
            episode_script = self.pre_sales.episode_generator(sample_goods_attr, sample_entity)
        elif episode == 'in_sales':
            desired_entity = self.sample_desired_entity()
            episode_script = self.in_sales.episode_generator(desired_entity)
        elif episode == 'after_sales':
            episode_script = self.after_sales.episode_generator()
        elif episode == 'pre_sales in_sales':
            sample_entity, sample_goods_attr, attr_priority, hard_constrains = self.sample_user()
            episode_script = self.pre_sales.episode_generator(sample_goods_attr, sample_entity)
            desired_entity = self.calculate_desired_entity(sample_entity, sample_goods_attr,
                                                           attr_priority, hard_constrains)
            episode_script.extend(self.in_sales.episode_generator(desired_entity))
        else:
            sys.exit("Unconsidered situations happen!")

        # And then, we translate some content based on KB results.

        # In the end, we add sentiment factor.

        return episode_script

    def decorate_sentiment(self, episode_script):
        pass

    def translate(self, episode_script):
        pass


if __name__ == '__main__':
    episode_generator = EpisodeGenerator(AVAILABLE_INTENT_5)

    # test our code
    for _ in range(10000):
        print('.......................\n')
        episode_script = episode_generator.episode_generator()
        for line in episode_script:
            for l in list(line.values())[0]:
                print(l)
            print('')
