# coding=utf-8
import json
from config import *
import os
import io


class DataAnalysis(object):
    def __init__(self, data_file):
        self.data_file = data_file

    def analysis(self):
        with open(os.path.join(DATA_ROOT, "public_1", self.data_file), "rb") as f:
            data = json.load(f)

        # todo: we can add more statistical factor!
        # Now we just want to know how many different agent response.
        results = {"agent_answer": None}
        agent_answer = list()
        for episode in data.values():
            for i, turn in enumerate(episode):
                if i % 2 == 1 and turn not in agent_answer:
                    agent_answer.append(turn)
        results["agent_answer"] = sorted(agent_answer)

        print(len(results["agent_answer"]))
        with io.open(os.path.join(DATA_ROOT, "statistics", self.data_file), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    for key in TASKS.keys():
        data_analysis = DataAnalysis(key + ".json")
        data_analysis.analysis()
