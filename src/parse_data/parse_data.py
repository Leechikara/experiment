# coding=utf-8

"""
This file parse data like babi-task
"""
from config import *
import json
import os
import io
import jieba
import re


class ParseData(object):
    def __init__(self, user_dict="specialSign.txt"):
        with open(os.path.join(DATA_ROOT, user_dict), "rb") as f:
            self.specialSign = f.readlines()

    @staticmethod
    def cut(l):
        special_sign = re.findall(r"\$\S+?\$", l)
        l = re.sub(r"\$\S+?\$", "PLACEHOLDER", l)
        cut_l = list()
        for word in jieba.cut(l):
            if word == "PLACEHOLDER":
                cut_l.append(special_sign[0])
                del special_sign[0]
            else:
                cut_l.append(word)
        return " ".join(cut_l)

    def get_vocab(self):
        for task in TASKS.keys():
            vocab = dict()
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "public_1"))

    def parse_candidate(self):
        for task in TASKS.keys():
            candidate = list()
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "statistics", data_set_name, task + ".json"), "rb") as f:
                    data = json.load(f)
                candidate.extend(data["agent_answer"])

            with io.open(os.path.join(DATA_ROOT, "candidate", task + ".txt"), "w", encoding="utf-8") as f:
                for line in set(candidate):
                    f.write(self.cut(line) + "\n")

    def parse_dialog(self):
        pass


if __name__ == "__main__":
    data_parser = ParseData()
    data_parser.parse_candidate()
