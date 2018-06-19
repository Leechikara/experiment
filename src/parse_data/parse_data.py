# coding=utf-8

"""
This file parse data like babi-task
"""

import json
import os
import io
import jieba
from config import *


class ParseData(object):
    @staticmethod
    def parse_candidate():
        for task in TASKS.keys():
            candidate = list()
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "statistics", data_set_name, task + ".json"), "rb") as f:
                    data = json.load(f)
                candidate.extend(data["agent_answer"])

            with io.open(os.path.join(DATA_ROOT, "candidate", task + ".json")) as f:
                for line in candidate:
                    f.write(line + "\n")

    @staticmethod
    def parse_dialog():
