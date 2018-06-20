# coding = utf-8
import numpy as np
import json
import os
from config import *
import re


class Vectorizer(object):
    def __init__(self, task):
        with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "rb") as f:
            vocab = json.load(f)
        self.word2index = vocab["word2index"]
        self.index2word = vocab["index2word"]

    def vocab_dim(self):
        return len(self.word2index)

    def vectorize_utt(self, utt):
        vec = np.zeros(len(self.word2index))
        for w in utt.split(' '):
            # Do not use special symbol features
            w = re.sub(r"<\S+?>", "", w)
            try:
                vec[self.word2index[w]] = 1
            except KeyError:
                vec[self.word2index["UNK"]] = 1
        return vec

    def vectorize_all(self, context_response_pairs):
        tensor = np.ndarray((len(context_response_pairs), 2, len(self.word2index)))

        for ind, context_response in enumerate(context_response_pairs):
            context, response = context_response
            context_vec = self.vectorize_utt(context)
            response_vec = self.vectorize_utt(response)
            tensor[ind][0] = context_vec
            tensor[ind][1] = response_vec

        return tensor

    @staticmethod
    def load_train(train_filename):
        context_response_pairs = []
        with open(train_filename, 'r') as f:
            for line in f:
                context, response = line.strip("\n").split('\t')
                context_response_pairs.append((context, response))
        return context_response_pairs

    def make_tensor(self, train_filename):
        train = self.load_train(train_filename)
        X = self.vectorize_all(train)
        print(train_filename, X.shape)
        return X
