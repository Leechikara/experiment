# coding = utf-8
import json, io, random, os, sys, re

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import *


class KnowledgeBase(object):
    def __init__(self, kb_file="kb.p"):
        self.kb_file = os.path.join(DATA_ROOT, kb_file)
        try:
            with open(self.kb_file, "rb") as f:
                self.kb = json.load(f)
        except IOError:
            self.kb = self.generate_original_kb()
            with io.open(self.kb_file, "w", encoding="utf-8") as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)

        # for evaluating our method when KB changes
        if CHANGE_ORIGINAL_ENTITY:
            # todo: change original entity
            pass
        if NEW_ADD_ENTITY > 0:
            self.expand_kb()
        if CHANGE_ORIGINAL_ENTITY or NEW_ADD_ENTITY > 0:
            with io.open(self.kb_file, "w", encoding="utf-8") as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)

    # We construct the kb below. We may modify kb dynamically.
    def fill_value(self, attr_name, entity_id, entity):
        if attr_name in entity.keys():
            return entity

        attr_definition = GOODS_ATTRIBUTE_DEFINITION[attr_name]

        # fill correlated attribute first
        if "correlate" in attr_definition.keys() and attr_definition["correlate"] not in entity.keys():
            found_correlate_attr = False
            for correlate_attr_name in GOODS_ATTRIBUTE_DEFINITION.keys():
                if correlate_attr_name == attr_definition["correlate"]:
                    found_correlate_attr = True
                    break
            assert found_correlate_attr, "Can't find correlated attribute!"
            entity = self.fill_value(correlate_attr_name, entity_id, entity)

        # fill value according to dtype
        if attr_definition["dtype"] == "float":
            if attr_definition.get("acceptNone", False) and random.random() < 0.5:
                value = None
            else:
                value = random.choice(attr_definition["range"])
            entity[attr_name] = round(value, 1) if value is not None else value
            return entity
        elif attr_definition["dtype"] == "int":
            if attr_definition.get("acceptNone", False) and random.random() < 0.5:
                value = None
            else:
                value = random.choice(attr_definition["range"])
            entity[attr_name] = value
            return entity
        elif attr_definition["dtype"] == "str":
            if "correlate" in attr_definition.keys() and entity[attr_definition["correlate"]] is None:
                value = None
            else:
                if attr_name == "discountURL":
                    value = attr_definition["prefix"] + "entity_id=" + str(entity_id) + "&" + "discountValue=" + str(
                        int(entity[attr_definition["correlate"]])) + attr_definition["postfix"]
                else:
                    value = random.choice(attr_definition["range"])
            entity[attr_name] = value
            return entity
        elif attr_definition["dtype"] == "list":
            amount = random.randint(1, len(attr_definition["range"]))
            value = random.sample(attr_definition["range"], amount)
            entity[attr_name] = value
            return entity
        elif attr_definition["dtype"] == "bool":
            value = random.choice(attr_definition["range"])
            entity[attr_name] = value
            return entity
        else:
            sys.exit("Unconsidered situations happened!")

    @staticmethod
    def make_price(entity):
        price_factor = 0
        factor_num = 0
        for attr_name, attr_definition in GOODS_ATTRIBUTE_DEFINITION.items():
            if "expensive" in attr_definition.keys():
                factor_num += 1
                if attr_definition["dtype"] in ["int", "float"]:
                    factor = (attr_definition["range"][-1] - entity[attr_name]) / (
                        attr_definition["range"][-1] - attr_definition["range"][0])
                else:
                    factor = 1 - (attr_definition["range"].index(entity[attr_name]) + 1) / len(attr_definition["range"])
                if attr_definition["expensive"] == "high":
                    factor = 1 - factor
                price_factor += factor

        price_factor /= factor_num
        max_price = GOODS_ATTRIBUTE_DEFINITION["price"]["max"]
        min_price = GOODS_ATTRIBUTE_DEFINITION["price"]["min"]
        entity["price"] = int(
            (max_price - min_price) * price_factor + min_price + (random.random() - 0.5) * PRICE_NOISE)
        return entity

    def generate_entity(self, entity_id):
        entity = {"id": entity_id}
        for attr_name in GOODS_ATTRIBUTE_DEFINITION.keys():
            if attr_name == "price":
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
        for i in range(NEW_ADD_ENTITY):
            entity_id = i + original_kb_size
            entity = self.generate_entity(entity_id)
            self.kb.append(entity)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # We define the kb operations below
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def find_entity(self, entity_id):
        """
        Find a entity by the id(type=int or entityId=xx)
        """
        if type(entity_id) == str:
            entity_id = int(re.findall(r"\d+", entity_id)[0])

        if type(entity_id) != int:
            sys.exit("Illegal data type of entity_id!")

        found_entity = False
        for entity in self.kb:
            if entity["id"] == entity_id:
                found_entity = True
                break
        assert found_entity, "Can't find entity given by users!"
        return entity

    def inform(self, attr_name, entity_list):
        """
        Inform user some information
        """
        assert 0 <= len(entity_list) < 3

        kb_results = {attr_name: None}
        if len(entity_list) != 0:
            kb_results[attr_name] = dict()
            for entityId in entity_list:
                entity = self.find_entity(entityId)
                kb_results[attr_name][entityId] = entity[attr_name]
        else:
            kb_results[attr_name] = OTHER_ATTRIBUTE_DEFINITION[attr_name]["value"]

        return kb_results

    def search_kb(self, **kwargs):
        """
        search KB based on some constrain conditions
        """
        candidates = list()
        if kwargs["dtype"] in ["int", "float"]:
            for entity in self.kb:
                if kwargs["compare"] == ">" and entity[kwargs["attr"]] > kwargs["value"]:
                    candidates.append(entity)
                elif kwargs["compare"] == "<" and entity[kwargs["attr"]] < kwargs["value"]:
                    candidates.append(entity)
                elif kwargs["compare"] == "=" and entity[kwargs["attr"]] == kwargs["value"]:
                    candidates.append(entity)
                elif kwargs["compare"] == "~" and kwargs["lower"] <= entity[kwargs["attr"]] <= kwargs["upper"]:
                    candidates.append(entity)
            return candidates
        elif kwargs["dtype"] in ["str", "bool"]:
            for entity in self.kb:
                if entity[kwargs["attr"]] == kwargs["value"]:
                    candidates.append(entity)
            return candidates
        elif kwargs["dtype"] in ["list"]:
            for entity in self.kb:
                if kwargs["value"] in entity[kwargs["attr"]]:
                    candidates.append(entity)
            return candidates
        else:
            pass


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.inform("color", "entityId=0", 1))
