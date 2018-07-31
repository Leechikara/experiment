# coding = utf-8
import numpy as np
import os, sys, json, re
from itertools import chain
from collections import defaultdict

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT, TASKS


class DataUtils(object):
    def __init__(self, ctx_encode):
        self.word2index = dict()
        self.index2word = dict()
        self.vocab_size = None

        self.candidates = list()
        self.candid2index = dict()
        self.index2candid = dict()
        self.cand_size = None
        self.all_data = dict()

        self.max_story_size = None
        self.mean_story_size = None
        self.sentence_size = None
        self.candidate_sentence_size = None
        self.query_size = None
        self.memory_size = None

        self.ctx_encode = ctx_encode

        # self.tagged = defaultdict(list)
        # self.comingS, self.comingQ, self.comingA = None, None, None

    def load_vocab(self):
        self.word2index = dict()
        self.index2word = dict()
        self.vocab_size = None
        vocab_set = set()

        for task in TASKS.keys():
            with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "rb") as f:
                vocab = json.load(f)
            vocab_set = set(vocab["word2index"].keys()) | vocab_set
        vocab = sorted(list(vocab_set)) + ["$u", "$r"] + ["#" + str(i) for i in range(30)]

        for i, w in enumerate(vocab):
            self.word2index[w] = i + 1
            self.index2word[i + 1] = w

        self.vocab_size = len(self.word2index) + 1

    def load_candidates(self):
        self.candidates = list()
        self.candid2index = dict()
        self.index2candid = dict()
        self.cand_size = None
        cand_set = set()

        for task in TASKS.keys():
            with open(os.path.join(DATA_ROOT, "candidate", task + ".txt")) as f:
                for line in f:
                    line = line.strip()
                    cand_set.add(line)

        cand_set = sorted(list(cand_set))
        for i, cand in enumerate(cand_set):
            self.candid2index[cand] = i
            self.index2candid[i] = cand
            self.candidates.append(cand.split())
        self.cand_size = len(self.candidates)

    def load_dialog(self):
        self.all_data = dict()
        for task in TASKS.keys():
            for data_set in ["train", "dev", "test"]:
                # In on-line learning, dev is a part of dev.
                data_file = os.path.join(DATA_ROOT, "public", task, data_set + ".txt")
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
                    for nid, description in enumerate(context[:-1]):
                        if nid % 2 == 0:
                            description.extend(["$u", "#" + str(nid // 2)])
                        else:
                            description.extend(["$r", "#" + str(nid // 2)])
                        memory.append(description)
                    query = context[-1]
                    # if train and test are not same, the response will be invalid
                    response = self.candid2index.get(_response, len(self.candid2index))
                    data.append((memory, query, response))

                self.all_data.setdefault(task, {})
                self.all_data[task][data_set] = data

    def build_pad_config(self, memory_size):
        data = list()
        for task in TASKS.keys():
            for data_set in ["train", "dev", "test"]:
                data.extend(self.all_data[task][data_set])

        self.max_story_size = max(map(len, (s for s, _, _ in data)))
        self.mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
        sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        self.candidate_sentence_size = max(map(len, self.candidates))
        self.query_size = max(map(len, (q for _, q, _ in data)))
        self.memory_size = min(memory_size, self.max_story_size)
        self.sentence_size = max(self.query_size, sentence_size)

        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
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
        queries = list()
        answers = list()
        data.sort(key=lambda x: len(x[0]), reverse=True)
        for i, (story, query, answer) in enumerate(data):
            if i % batch_size == 0:
                memory_size = max(1, min(self.memory_size, len(story)))
                if self.ctx_encode != "MemoryNetwork":
                    memory_size += 1
            ss = []
            if self.ctx_encode != "MemoryNetwork":
                story = story.append(query)
            for sentence in story:
                ls = max(0, self.sentence_size - len(sentence))
                ss.append([self.word2index[w] if w in self.word2index else self.word2index["UNK"] for w in sentence] + [
                    0] * ls)

            # take only the most recent sentences that fit in memory
            ss = ss[::-1][:memory_size][::-1]

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * self.sentence_size)

            lq = max(0, self.sentence_size - len(query))
            q = [self.word2index[w] if w in self.word2index else self.word2index["UNK"] for w in query] + [0] * lq

            stories.append(np.array(ss, dtype=np.int64))
            queries.append(np.array(q, dtype=np.int64))
            answers.append(np.array(answer, dtype=np.int64))
        return stories, queries, answers

    # def sample_true(self, resp_c, sample_n):
    #     if len(self.tagged[resp_c]) < sample_n:
    #         return list(np.random.choice(np.asarray(self.tagged[resp_c]), sample_n, replace=True))
    #     else:
    #         return list(np.random.choice(np.asarray(self.tagged[resp_c]), sample_n, replace=False))
    #
    # def sample_negative(self, resp_c, sample_n):
    #     negative_cand = list()
    #     for k, v in self.tagged.items():
    #         if k == resp_c:
    #             continue
    #         negative_cand.extend(v)
    #     if len(negative_cand) < sample_n:
    #         return list(np.random.choice(np.asarray(negative_cand), sample_n, replace=True))
    #     else:
    #         return list(np.random.choice(np.asarray(negative_cand), sample_n, replace=False))


def batch_iter(stories, queries, answers, batch_size, shuffle=False):
    data_num = len(stories)
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
        stories_batch = stories[start:end]
        queries_batch = queries[start:end]
        if answers is None:
            answers_batch = None
        else:
            answers_batch = answers[start:end]
        yield stories_batch, queries_batch, answers_batch, start
