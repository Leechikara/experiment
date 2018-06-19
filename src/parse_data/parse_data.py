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
    def __init__(self, user_dict="specialSign.json"):
        with open(os.path.join(DATA_ROOT, user_dict), "rb") as f:
            self.specialSign = json.load(f)

    def cut(self, l):
        special_sign = list(re.findall(r"\$\S+?\$", l))
        l = re.sub(r"\$\S+?\$", "PLACEHOLDER", l)
        cut_l = list()
        for word in jieba.cut(l):
            if word == "PLACEHOLDER":
                cut_l.append(self.specialSign[special_sign[0]])
                del special_sign[0]
            else:
                cut_l.append(word)
        return " ".join(cut_l)

    def get_vocab(self):
        for task in TASKS.keys():
            vocab = set()
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "public_1", task, data_set_name + ".json"), 'rb') as f:
                    data = json.load(f)
                for meta_data in data.values():
                    for line in meta_data["episode_content"]:
                        line = self.cut(line)
                        for word in line.split():
                            vocab.add(word)
            vocab = list(vocab)
            word2index = dict()
            index2word = dict()
            for i, word in enumerate(vocab):
                word2index[word] = i
                index2word[i] = word
            with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "w", encoding="utf-8") as f:
                json.dump({"word2index": word2index, "index2word": index2word}, f, ensure_ascii=False, indent=2)

    def parse_candidate(self):
        for task in TASKS.keys():
            candidate = list()
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "statistics", task, data_set_name + ".json"), "rb") as f:
                    data = json.load(f)
                candidate.extend(data["agent_answer"])

            with io.open(os.path.join(DATA_ROOT, "candidate", task + ".txt"), "w", encoding="utf-8") as f:
                for line in set(candidate):
                    f.write(self.cut(line) + "\n")

    def parse_dialog(self):
        pass


if __name__ == "__main__":
    data_parser = ParseData()
    # data_parser.parse_candidate()
    data_parser.get_vocab()
