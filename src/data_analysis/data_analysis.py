# coding=utf-8
import json
from config import *
import os
import io


class DataAnalysis(object):
    @staticmethod
    def analysis():
        for task in TASKS.keys():
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "public_1", task, data_set_name + ".json"), "rb") as f:
                    data = json.load(f)

                # todo: we can add more statistical factor!
                # Now we just want to know how many different agent response.
                results = {"agent_answer": None}
                agent_answer = list()
                for meta_data in data.values():
                    for i, turn in enumerate(meta_data["episode_content"]):
                        if i % 2 == 1 and turn not in agent_answer:
                            agent_answer.append(turn)
                results["agent_answer"] = sorted(agent_answer)

                print(str(len(results["agent_answer"])) + " system actions we found in " + task + " " + data_set_name)
                with io.open(os.path.join(DATA_ROOT, "statistics", task, data_set_name + ".json"), "w",
                             encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_analysis = DataAnalysis()
    data_analysis.analysis()
