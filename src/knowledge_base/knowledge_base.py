# coding = utf-8
import json, io, random, os
from config import *


class KnowledgeBase(object):
    def __init__(self, kb_file='kb.p'):
        self.kb_file = os.path.join(DATA_ROOT, kb_file)
        try:
            with open(self.kb_file, 'rb') as f:
                self.kb = json.load(f)
        except IOError:
            self.kb = self.generate_original_kb()
            with io.open(self.kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)

        # for evaluating our method when KB changes
        if CHANGE_ORIGINAL_ENTITY:
            # todo: change original entity
            pass
        if ADD_NEW_ENTITY > 0:
            self.expand_kb()
        if CHANGE_ORIGINAL_ENTITY or ADD_NEW_ENTITY > 0:
            with io.open(self.kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)

    def fill_value(self, attr_name, entity_id, entity):
        if attr_name in entity.keys():
            return entity

        attr_definition = GOODS_ATTRIBUTE_DEFINITION[attr_name]

        # fill correlated attribute first
        if 'correlate' in attr_definition.keys() and attr_definition['correlate'] not in entity.keys():
            found_correlate_attr = False
            for correlate_attr_name in GOODS_ATTRIBUTE_DEFINITION.keys():
                if correlate_attr_name == attr_definition['correlate']:
                    found_correlate_attr = True
                    break
            assert found_correlate_attr, "Can't find correlated attribute!"
            entity = self.fill_value(correlate_attr_name, entity_id, entity)

        # fill value according to dtype
        if attr_definition['dtype'] == 'float':
            if attr_definition.get('acceptNone', False) and random.random() < 0.5:
                value = None
            else:
                value = random.choice(attr_definition['range'])
            entity[attr_name] = round(value, 1) if value is not None else value
            return entity
        elif attr_definition['dtype'] == 'int':
            value = random.choice(attr_definition['range'])
            entity[attr_name] = value
            return entity
        elif attr_definition['dtype'] == 'str':
            if 'correlate' in attr_definition.keys() and entity[attr_definition['correlate']] is None:
                value = None
            else:
                if attr_name == 'discountURL':
                    value = attr_definition['prefix'] + 'entity_id=' + str(entity_id) + '&' + 'discountValue=' + str(
                        int(entity[attr_definition['correlate']] * 10)) + attr_definition['postfix']
                else:
                    value = random.choice(attr_definition['range'])
            entity[attr_name] = value
            return entity
        elif attr_definition['dtype'] == 'list':
            amount = random.randint(1, len(attr_definition['range']))
            value = random.sample(attr_definition['range'], amount)
            entity[attr_name] = value
            return entity
        elif attr_definition['dtype'] == 'bool':
            value = random.choice(attr_definition['range'])
            entity[attr_name] = value
            return entity
        else:
            assert True is False, "Unconsidered situations happened!"

    @staticmethod
    def make_price(entity):
        price_factor = 0
        factor_num = 0
        for attr_name, attr_definition in GOODS_ATTRIBUTE_DEFINITION.items():
            if 'expensive' in attr_definition.keys():
                factor_num += 1
                if attr_definition['dtype'] in ['int', 'float']:
                    factor = (attr_definition['range'][-1] - entity[attr_name]) / (
                        attr_definition['range'][-1] - attr_definition['range'][0])
                else:
                    factor = 1 - (attr_definition['range'].index(entity[attr_name]) + 1) / len(attr_definition['range'])
                if attr_definition['expensive'] == 'high':
                    factor = 1 - factor
                price_factor += factor

        price_factor /= factor_num
        max_price = GOODS_ATTRIBUTE_DEFINITION['price']['max']
        min_price = GOODS_ATTRIBUTE_DEFINITION['price']['min']
        entity['price'] = int(
            (max_price - min_price) * price_factor + min_price + (random.random() - 0.5) * PRICE_NOISE)
        return entity

    def generate_entity(self, entity_id):
        entity = {'id': entity_id}
        for attr_name in GOODS_ATTRIBUTE_DEFINITION.keys():
            if attr_name == 'price':
                continue
            else:
                entity = self.fill_value(attr_name, entity_id, entity)
        entity = self.make_price(entity)
        return entity

    def generate_original_kb(self):
        kb = list()
        for entity_id in range(ORIGINAL_ENTITY):
            entity = self.generate_entity(entity_id)
            kb.append(entity)
        return kb

    def expand_kb(self):
        original_kb_size = len(self.kb)
        for i in range(ADD_NEW_ENTITY):
            entity_id = i + original_kb_size
            entity = self.generate_entity(entity_id)
            self.kb.append(entity)

    def find_entity(self, entity_id):
        found_entity = False
        for entity in self.kb:
            if entity['id'] == entity_id:
                found_entity = True
                break
        assert found_entity, "Can't find entity given by users!"
        return entity

    def inform(self, entity_id, attr_name):
        if entity_id is not None:
            entity = self.find_entity(entity_id)
            return entity[attr_name]
        else:
            attr_definition = OTHER_ATTRIBUTE_DEFINITION[attr_name]
            return attr_definition['value']

    def confirm(self, entity_id, attr_name, compare, value):
        entity = self.find_entity(entity_id)
        true_value = entity[attr_name]
        if type(true_value) in [float, int]:
            if true_value < value and compare == '<' or \
                                    true_value == value and compare == '=' or \
                                    true_value > value and compare == '>':
                return True
            else:
                return False
        elif type(true_value) == list:
            if value in true_value and compare == 'has':
                return True
            else:
                return False
        elif type(true_value) in [str, bool]:
            if true_value == value and compare == 'is':
                return True
            else:
                return False
        else:
            assert True is False, "Unconsidered situations happened!"

    def compare(self, entity1_id, entity2_id, attr_name):
        entity1 = self.find_entity(entity1_id)
        entity2 = self.find_entity(entity2_id)
        assert attr_name in GOODS_ATTRIBUTE_DEFINITION.keys(), "Can't find this attribute!"
        assert 'prefer' in GOODS_ATTRIBUTE_DEFINITION[attr_name].keys(), "This attribute is not comparable!"

        if type(entity1[attr_name]) == type(entity2[attr_name]) and type(entity1[attr_name]) in [float, int]:
            if GOODS_ATTRIBUTE_DEFINITION[attr_name]['prefer'] == 'low':
                if entity1[attr_name] < entity2[attr_name]:
                    return True, entity1_id, entity2_id
                elif entity1[attr_name] == entity2[attr_name]:
                    return False, entity1_id, entity2_id
                else:
                    return True, entity2_id, entity1_id
            else:
                if entity1[attr_name] < entity2[attr_name]:
                    return True, entity2_id, entity1_id
                elif entity1[attr_name] == entity2[attr_name]:
                    return False, entity1_id, entity2_id
                else:
                    return True, entity1_id, entity2_id
        elif type(entity1[attr_name]) == type(entity2[attr_name]) and type(entity1[attr_name]) in [str, bool]:
            if GOODS_ATTRIBUTE_DEFINITION[attr_name]['prefer'] == 'low':
                if GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity1[attr_name]) < \
                        GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity2[attr_name]):
                    return True, entity1_id, entity2_id
                elif GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity1[attr_name]) == \
                        GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity2[attr_name]):
                    return False, entity1_id, entity2_id
                else:
                    return True, entity2_id, entity1_id
            else:
                if GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity1[attr_name]) < \
                        GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity2[attr_name]):
                    return True, entity2_id, entity1_id
                elif GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity1[attr_name]) == \
                        GOODS_ATTRIBUTE_DEFINITION[attr_name]['range'].index(entity2[attr_name]):
                    return False, entity1_id, entity2_id
                else:
                    return True, entity1_id, entity2_id
        else:
            assert True is False, "Unconsidered situations happened!"
