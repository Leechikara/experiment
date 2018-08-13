# coding = utf-8
import numpy as np
import os, sys, json, re
from itertools import chain

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT


class DataUtils(object):
    def __init__(self):
        self.word2index = dict()
        self.index2word = dict()
        self.vocab_size = None

        self.candidates = list()
        self.candid2index = dict()
        self.index2candid = dict()
        self.cand_size = None

        self.max_story_size = None
        self.mean_story_size = None
        self.sentence_size = None
        self.candidate_sentence_size = None
        self.query_size = None
        self.memory_size = None

    def load_vocab(self, task):
        self.word2index = dict()
        self.index2word = dict()
        self.vocab_size = None

        with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "rb") as f:
            vocab = json.load(f)

        for key, value in vocab["word2index"].items():
            self.word2index[key] = int(value) + 1
        for key, value in vocab["index2word"].items():
            self.index2word[int(key) + 1] = value

        # add $u $r and #i
        new_vocab = ["$u", "$r"] + ["#" + str(i) for i in range(30)]
        for w in new_vocab:
            self.word2index[w] = len(self.word2index) + 1
            self.index2word[len(self.word2index)] = w

        self.vocab_size = len(self.word2index) + 1

    def load_candidates(self, task):
        self.candidates = list()
        self.candid2index = dict()
        self.index2candid = dict()
        self.cand_size = None

        with open(os.path.join(DATA_ROOT, "candidate", task + ".txt")) as f:
            for i, line in enumerate(f):
                line = line.strip()
                self.candid2index[line] = i
                self.index2candid[i] = line
                line = line.split()
                self.candidates.append(line)
        self.cand_size = len(self.candidates)

    def load_dialog(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()

        data = list()
        for line in lines:
            line = line.strip()
            _context, _response = line.split("\t")

            context = list()
            speaker = "user"
            description = list()
            for w in _context.split(" "):
                if w.find("agent") != -1 and speaker == "user":
                    speaker = "agent"
                    context.append(description)
                    description = list()
                elif w.find("user") != -1 and speaker == "agent":
                    speaker = "user"
                    context.append(description)
                    description = list()
                description.append(re.sub(r"<\S+?>", "", w))
            context.append(description)

            memory = list()
            for nid, description in enumerate(context):
                if nid % 2 == 0:
                    description.extend(["$u", "#" + str(nid // 2)])
                else:
                    description.extend(["$r", "#" + str(nid // 2)])
                memory.extend(description)
            # if train and test are not same, the response will be invalid
            response = self.candid2index.get(_response, len(self.candid2index))
            data.append((memory, response))

        return data

    def build_pad_config(self, data):
        self.max_story_size = max(map(len, (s for s, _ in data)))
        self.mean_story_size = int(np.mean([len(s) for s, _ in data]))
        self.candidate_sentence_size = max(map(len, self.candidates))

        print("vocab size:", self.vocab_size)
        print("Longest candidate sentence length", self.candidate_sentence_size)
        print("Longest story length", self.max_story_size)
        print("Average story length", self.mean_story_size)

    def vectorize_candidates(self):
        candidate_rep = list()
        for candidate in self.candidates:
            lc = max(0, self.candidate_sentence_size - len(candidate))
            candidate_rep.append(
                [self.word2index[w] if w in self.word2index else self.word2index["UNK"] for w in candidate] + [0] * lc)
        return np.asarray(candidate_rep, dtype=np.int64)

    def vectorize_data(self, data, batch_size):
        stories = list()
        answers = list()
        data.sort(key=lambda x: len(x[0]), reverse=True)
        for i, (story, _, answer) in enumerate(data):
            if i % batch_size == 0:
                memory_size = len(story)
            ls = memory_size - len(story)
            ss = [self.word2index[w] if w in self.word2index else self.word2index["UNK"] for w in story] + [0] * ls
            stories.append(np.array(ss, dtype=np.int64))
            answers.append(np.array(answer, dtype=np.int64))
        return stories, answers


def batch_iter(ctx, response, batch_size, shuffle=False):
    data_num = len(ctx)
    batches = zip(range(0, data_num - batch_size, batch_size),
                  range(batch_size, data_num, batch_size))
    extra_data_num = data_num % batch_size
    if extra_data_num > 0:
        batches = [(start, end) for start, end in batches] + [(data_num - extra_data_num, data_num)]
    else:
        batches = [(start, end) for start, end in batches] + [(data_num - batch_size, data_num)]

    if shuffle:
        np.random.shuffle(batches)

    for start, end in batches:
        stories_batch = ctx[start:end]
        if response is None:
            response_batch = None
        else:
            response_batch = response[start:end]
        yield stories_batch, response_batch


def build_p_mapping(source_s2ind, target_s2ind):
    mapping = list()
    for s in source_s2ind.keys():
        if s in target_s2ind:
            mapping.append((source_s2ind[s], target_s2ind[s]))
    return mapping
