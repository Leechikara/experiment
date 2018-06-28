# coding = utf-8
import numpy as np
import json, re, os, sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT


class Vectorizer(object):
    @staticmethod
    def load_vocab(task):
        with open(os.path.join(DATA_ROOT, "vocab", task + ".json"), "rb") as f:
            vocab = json.load(f)

        word2index = vocab["word2index"]
        index2word = vocab["index2word"]

        return word2index, index2word, len(word2index)

    @staticmethod
    def vectorize_utt(utt, word2index):
        vec = np.zeros(len(word2index), dtype=np.float32)
        for w in utt.split():
            # Do not use special symbol features
            w = re.sub(r"<\S+?>", "", w)
            try:
                vec[word2index[w]] = 1
            except KeyError:
                vec[word2index["UNK"]] = 1
        return vec

    def vectorize_all(self, context_response_pairs, word2index):
        tensor = np.ndarray((len(context_response_pairs), 2, len(word2index)), dtype=np.float32)

        for ind, context_response in enumerate(context_response_pairs):
            context, response = context_response
            context_vec = self.vectorize_utt(context, word2index)
            response_vec = self.vectorize_utt(response, word2index)
            tensor[ind][0] = context_vec
            tensor[ind][1] = response_vec

        return tensor

    @staticmethod
    def load_train(train_filename):
        context_response_pairs = []
        with open(train_filename, "r") as f:
            for line in f:
                context, response = line.strip("\n").split("\t")
                context_response_pairs.append((context, response))
        return context_response_pairs

    def make_tensor(self, train_filename, word2index):
        train = self.load_train(train_filename)
        X = self.vectorize_all(train, word2index)
        print(train_filename, X.shape)
        return X


def batch_iter(tensor, batch_size, shuffle=False):
    while tensor.shape[0] % batch_size > 0:
        extra_data_num = batch_size - tensor.shape[0] % batch_size
        extra_tensor = np.random.permutation(tensor)[:extra_data_num]
        tensor = np.append(tensor, extra_tensor, axis=0)
    batches_count = tensor.shape[0] // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
        data = tensor[shuffle_indices]
    else:
        data = tensor

    for batch_num in range(batches_count):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        yield data[start_index:end_index]


def neg_sampling_iter(tensor, batch_size, count, seed=None):
    while tensor.shape[0] % batch_size > 0:
        extra_data_num = batch_size - tensor.shape[0] % batch_size
        extra_tensor = np.random.permutation(tensor)[:extra_data_num]
        tensor = np.append(tensor, extra_tensor, axis=0)
    batches_count = tensor.shape[0] // batch_size

    trials = 0
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    data = tensor[shuffle_indices]
    for batch_num in range(batches_count):
        trials += 1
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        if trials > count:
            return
        else:
            yield data[start_index:end_index]


def build_p_mapping(source_s2ind, target_s2ind):
    mapping = list()
    for s in source_s2ind.keys():
        if s in target_s2ind:
            mapping.append((source_s2ind[s], target_s2ind[s]))
    return mapping
