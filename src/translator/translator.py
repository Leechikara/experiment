# coding = utf-8
"""
We translate agent reply into NL.
"""
import json
import os
from config import *
from src.knowledge_base.knowledge_base import KnowledgeBase
import copy
import re


class Translator(object):
    def __init__(self, nl_pairs_file='nl_pairs.json'):
        with open(os.path.join(DATA_ROOT, nl_pairs_file), 'rb') as f:
            self.nl_pairs = json.load(f)
        self.kb_helper = KnowledgeBase()

    def translate(self, agent_action):
        """
        Translate an agent action into a NL
        """
        # Some agent actions do not need translate.
        if not agent_action[0] == '$' and agent_action[-1] == '$':
            return agent_action

        # Get the basic key words to query KB.
        agent_action = agent_action.strip('$')
        items = agent_action.split("_")
        kb_query_keys = dict()
        order = 0
        entity_list = list()
        for item in items:
            if item.find("entityId=") != -1:
                order += 1
                kb_query_keys['entity' + str(order)] = item
                entity_list.append(item)
            else:
                kb_query_keys['attr'] = item

        # Get the results from KB
        attr = kb_query_keys['attr']
        kb_results = self.kb_helper.inform(attr, entity_list)

        if len(entity_list) == 0:
            nl = copy.deepcopy(self.nl_pairs[attr])
            # Substitute KB results
            information = kb_results[attr]
            if type(information) == list:
                information = ",".join(information)
            else:
                information = str(information)
            nl = re.sub(r"\$\S+\$", information, nl)
        elif len(entity_list) == 1:
            if kb_results[attr][kb_query_keys['entity1']] is None:
                nl = copy.deepcopy(self.nl_pairs['_'.join(['entity', attr])]['None'])
            elif kb_results[attr][kb_query_keys['entity1']] is True:
                nl = copy.deepcopy(self.nl_pairs['_'.join(['entity', attr])]['True'])
            elif kb_results[attr][kb_query_keys['entity1']] is False:
                nl = copy.deepcopy(self.nl_pairs['_'.join(['entity', attr])]['False'])
            elif type(self.nl_pairs['_'.join(['entity', attr])]) == dict:
                nl = copy.deepcopy(self.nl_pairs['_'.join(['entity', attr])]['notNone'])
            else:
                nl = copy.deepcopy(self.nl_pairs['_'.join(['entity', attr])])
            # Substitute entity
            nl = re.sub(r"\$entity\$", '$' + kb_query_keys['entity1'] + '$', nl)
            # Substitute KB results
            information = kb_results[attr][kb_query_keys['entity1']]
            if type(information) == list:
                information = ",".join(information)
            else:
                information = str(information)
            nl = re.sub(r"\$entity_\S+\$", information, nl)
        else:
            if kb_results[attr][kb_query_keys['entity1']] == kb_results[attr][kb_query_keys['entity2']]:
                if type(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['same']) == str:
                    nl = copy.deepcopy(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['same'])
                    # Substitute entity
                    nl = re.sub(r"\$entity1\$", '$' + kb_query_keys['entity1'] + '$', nl)
                    nl = re.sub(r"\$entity2\$", '$' + kb_query_keys['entity2'] + '$', nl)
                    # Substitute KB results
                    information = kb_results[attr][kb_query_keys['entity1']]
                    if type(information) == list:
                        information = ",".join(information)
                    else:
                        information = str(information)
                    nl = re.sub(r"\$entity\d_\S+\$", information, nl)
                else:
                    if kb_results[attr][kb_query_keys['entity1']] is None:
                        nl = copy.deepcopy(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['same']['None'])
                    else:
                        nl = copy.deepcopy(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['same']['notNone'])
                    # Substitute entity
                    nl = re.sub(r"\$entity1\$", '$' + kb_query_keys['entity1'] + '$', nl)
                    nl = re.sub(r"\$entity2\$", '$' + kb_query_keys['entity2'] + '$', nl)
                    # Substitute KB results
                    if kb_results[attr][kb_query_keys['entity1']] is not None:
                        information = kb_results[attr][kb_query_keys['entity1']]
                        if type(information) == list:
                            information = ",".join(information)
                        else:
                            information = str(information)
                        nl = re.sub(r"\$entity\d_\S+\$", information, nl)
            else:
                if type(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['notSame']) == str:
                    nl = copy.deepcopy(self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['notSame'])
                    # Substitute entity
                    nl = re.sub(r"\$entity1\$", '$' + kb_query_keys['entity1'] + '$', nl)
                    nl = re.sub(r"\$entity2\$", '$' + kb_query_keys['entity2'] + '$', nl)
                    # Substitute KB results
                    information = kb_results[attr][kb_query_keys['entity1']]
                    if type(information) == list:
                        information = ",".join(information)
                    else:
                        information = str(information)
                    nl = re.sub(r"\$entity1_\S+?\$", information, nl)
                    information = kb_results[attr][kb_query_keys['entity2']]
                    if type(information) == list:
                        information = ",".join(information)
                    else:
                        information = str(information)
                    nl = re.sub(r"\$entity2_\S+?\$", information, nl)
                else:
                    if kb_results[attr][kb_query_keys['entity1']] is not None and \
                                    kb_results[attr][kb_query_keys['entity2']] is not None:
                        nl = copy.deepcopy(
                            self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['notSame']['notNone_notNone'])
                    elif kb_results[attr][kb_query_keys['entity1']] is not None and \
                                    kb_results[attr][kb_query_keys['entity2']] is None:
                        nl = copy.deepcopy(
                            self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['notSame']['notNone_None'])
                    else:
                        nl = copy.deepcopy(
                            self.nl_pairs['_'.join(['entity1', 'entity2', attr])]['notSame']['None_notNone'])
                    # Substitute entity
                    nl = re.sub(r"\$entity1\$", '$' + kb_query_keys['entity1'] + '$', nl)
                    nl = re.sub(r"\$entity2\$", '$' + kb_query_keys['entity2'] + '$', nl)
                    # Substitute KB results
                    if kb_results[attr][kb_query_keys['entity1']] is not None:
                        information = kb_results[attr][kb_query_keys['entity1']]
                        if type(information) == list:
                            information = ",".join(information)
                        else:
                            information = str(information)
                        nl = re.sub(r"\$entity1_\S+?\$", information, nl)
                    if kb_results[attr][kb_query_keys['entity2']] is not None:
                        information = kb_results[attr][kb_query_keys['entity2']]
                        if type(information) == list:
                            information = ",".join(information)
                        else:
                            information = str(information)
                        nl = re.sub(r"\$entity2_\S+?\$", information, nl)
        return nl


if __name__ == "__main__":
    translator = Translator()
    print(translator.translate('$entityId=1_entityId=2_price$'))
