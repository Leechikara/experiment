# coding=utf-8

"""
This file parse data like babi-task
"""
import json, re, os, io, jieba, sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT, DATA_SET, TASKS


class ParseData(object):
    def __init__(self, user_dict="specialSign.json"):
        # parse special symbol like $*$
        with open(os.path.join(DATA_ROOT, user_dict), "rb") as f:
            self.specialSign = json.load(f)

    def cut(self, l):
        # cut a sentence with special symbol
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
            vocab = sorted(list(vocab))
            word2index = dict()
            index2word = dict()
            for i, word in enumerate(vocab):
                word2index[word] = i
                index2word[i] = word
            # add UNK
            word2index["UNK"] = i + 1
            index2word[i + 1] = "UNK"
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
                for line in sorted(list(set(candidate))):
                    f.write(self.cut(line) + "\t\n")

    @staticmethod
    def decorate_line(line, role):
        # add some feature after cut a line
        # we add speaker role, digital and entity tag in the form of one-hot feature
        decorated_line = list()
        for word in line.split():
            decorated_word = word
            if word.find("entity") != -1:
                decorated_word += "<entity>"
            if re.match(r"^\d+(\.\d{1,2})?", word) is not None:
                decorated_word += "<digital>"
            decorated_word += role
            decorated_line.append(decorated_word)
        decorated_line = " ".join(decorated_line)
        return decorated_line

    def parse_dialog(self):
        for task in TASKS.keys():
            for data_set_name in DATA_SET.keys():
                with open(os.path.join(DATA_ROOT, "public_1", task, data_set_name + ".json"), "r") as f:
                    data = json.load(f)

                dialogs = list()
                for meta_data in data.values():
                    dialog = list()
                    assert len(meta_data["episode_content"]) % 2 == 0
                    for turn in range(len(meta_data["episode_content"]) // 2):
                        user_utt = self.cut(meta_data["episode_content"][2 * turn])
                        user_utt = self.decorate_line(user_utt, "<user>")
                        agent_utt = self.cut(meta_data["episode_content"][2 * turn + 1])
                        if len(dialog) != 0:
                            prev_step = dialog[-1]
                            user_utt_with_history = "{} {} {}".format(prev_step[1],
                                                                      self.decorate_line(prev_step[2], "<agent>"),
                                                                      user_utt)
                        else:
                            user_utt_with_history = user_utt
                        dialog.append((turn, user_utt_with_history, agent_utt))
                    dialogs.append(dialog)
                with io.open(os.path.join(DATA_ROOT, "public", task, data_set_name + ".txt"), "w",
                             encoding="utf-8") as f:
                    for dialog in dialogs:
                        for _, user_utt, agent_utt in dialog:
                            f.write('{}\t{}\n'.format(user_utt, agent_utt))
                        f.write("\t")


if __name__ == "__main__":
    data_parser = ParseData()
    data_parser.parse_candidate()
    data_parser.get_vocab()
    data_parser.parse_dialog()
