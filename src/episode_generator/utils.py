# coding = utf-8
import json
import io
import pickle
import random
from config import *
import os


class KnowledgeBase(object):
    def __init__(self, kb_file='kb.p'):
        self.kb_file = os.path.join(DATA_ROOT, kb_file)
        try:
            with open(self.kb_file, 'rb') as f:
                self.kb = json.load(f)
        except IOError:
            self.kb = self.generate_kb()
            with io.open(self.kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)
        if NEW_ENTITY > 0:
            self.expand_kb()
            with io.open(self.kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)

    def fill_value(self, attr_index, entity_id, entity):
        attr = GOODS_ATTRIBUTE_DEFINITION[attr_index]
        if attr['key'] in entity.keys():
            return entity

        # fill correlated attribute first
        if 'correlate' in attr.keys() and attr['correlate'] not in entity.keys():
            found_correlate_attr = False
            for correlate_attr in GOODS_ATTRIBUTE_DEFINITION:
                if correlate_attr['key'] == attr['correlate']:
                    found_correlate_attr = True
                    break
            assert found_correlate_attr, "Can't find correlated attribute!"
            correlate_index = correlate_attr['index']
            entity = self.fill_value(correlate_index, entity_id, entity)

        # fill value according to dtype
        if attr['dtype'] == 'float':
            if attr.get('acceptNone', False) and random.random() < 0.5:
                value = None
            else:
                value = random.choice(attr['range'])
            entity[attr['key']] = round(value, 1) if value is not None else value
            return entity
        elif attr['dtype'] == 'str':
            if 'correlate' in attr.keys() and entity[attr['correlate']] is None:
                value = None
            else:
                if 'prefix' in attr.keys() and 'postfix' in attr.keys():
                    value = attr['prefix'] + 'entity_id=' + str(entity_id) + '&' + 'discountValue=' + str(
                        int(entity[attr['correlate']] * 10)) + attr['postfix']
                else:
                    value = random.choice(attr['range'])
            entity[attr['key']] = value
            return entity
        elif attr['dtype'] == 'list':
            amount = random.randint(1, len(attr['range']))
            value = random.sample(attr['range'], amount)
            entity[attr['key']] = value
            return entity
        elif attr['dtype'] == 'bool':
            value = random.choice(attr['range'])
            entity[attr['key']] = value
            return entity
        elif attr['dtype'] == 'int':
            value = random.choice(attr['range'])
            entity[attr['key']] = value
            return entity
        else:
            assert True is False, "Unconsidered situations happened!"

    @staticmethod
    def make_price(entity):
        price_factor = 0
        factor_num = 0
        for attr in GOODS_ATTRIBUTE_DEFINITION:
            if 'expensive' in attr.keys():
                factor_num += 1
                if attr['dtype'] in ['int', 'float']:
                    factor = (attr['range'][-1] - entity[attr['key']]) / (attr['range'][-1] - attr['range'][0])
                else:
                    factor = attr['range'].index(entity[attr['key']]) / len(attr['range'])
                if attr['expensive'] == 'high':
                    factor = 1 - factor
                price_factor += factor

        price_factor /= factor_num
        max_price = GOODS_ATTRIBUTE_DEFINITION[0]['max']
        min_price = GOODS_ATTRIBUTE_DEFINITION[0]['min']
        entity['price'] = int(
            (max_price - min_price) * price_factor + min_price + (random.random() - 0.5) * PRICE_NOISE)
        return entity

    def generate_kb(self):
        kb = list()
        for entity_id in range(MAX_ENTITY):
            entity = {'id': entity_id}
            for attr_index, attr in enumerate(GOODS_ATTRIBUTE_DEFINITION):
                if attr['key'] == 'price':
                    continue
                else:
                    entity = self.fill_value(attr_index, entity_id, entity)
            entity = self.make_price(entity)
            kb.append(entity)
        return kb

    def expand_kb(self):
        old_entity_amount = len(self.kb)
        for i in range(NEW_ENTITY):
            entity_id = i + old_entity_amount
            entity = {'id': entity_id}
            for attr_index, attr in enumerate(GOODS_ATTRIBUTE_DEFINITION):
                if attr['key'] == 'price':
                    continue
                else:
                    entity = self.fill_value(attr_index, entity_id, entity)
            entity = self.make_price(entity)
            self.kb.append(entity)


kb = KnowledgeBase()
