# coding = utf-8
import numpy as np
import json, os, sys, re

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import *


class Vectorizer(object):
    def __init__(self, task):
        with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "rb") as f:
            vocab = json.load(f)
        self.word2index = vocab["word2index"]
        self.index2word = vocab["index2word"]
        with open(os.path.join(DATA_ROOT, "feature.json"), "rb") as f:
            self.feature_dict = json.load(f)

    def vectorize_contexts(self, contexts):
        word_tensor = np.ndarray((len(contexts), len(self.word2index)))
        feature_tensor = np.ndarray((len(contexts), len(self.feature_dict)))
        for ind, context in enumerate(contexts):
            vec_word = np.zeros(len(self.word2index))
            vec_feature = np.zeros(len(self.feature_dict))
            for w in context.split():
                word = re.sub(r"<\S+?>", "", w)
                vec_word[self.word2index[word]] = 1
                for feature in re.findall(r"<\S+?>", w):
                    vec_feature[self.feature_dict[feature]] = 1
            word_tensor[ind] = vec_word
            feature_tensor[ind] = vec_feature
        return np.concatenate((word_tensor, feature_tensor), axis=1)

    def vectorize_responses(self, responses):
        word_tensor = np.ndarray((len(responses), len(self.word2index)))
        for ind, response in enumerate(responses):
            vec = np.zeros(len(self.word2index))
            for w in response.split():
                vec[self.word2index[w]] = 1
            word_tensor[ind] = vec
        return word_tensor

    def vectorize_all(self, train, vocab):
        contexts = [item[0] for item in train]
        responses = [item[1] for item in train]
        context_tensor = self.vectorize_contexts(contexts)
        response_tensor = self.vectorize_responses(responses)
        return np.concatenate((context_tensor, response_tensor), axis=1)

    @staticmethod
    def load_train(train_filename):
        context_response_pairs = []
        with open(train_filename, 'r') as f:
            for line in f:
                context, response = line.strip().split('\t')
                context_response_pairs.append((context, response))
        return context_response_pairs

    def make_tensor(self, train_filename, vocab):
        train = self.load_train(train_filename)
        X = self.vectorize_all(train, vocab)
        print(train_filename, X.shape)
        return X
