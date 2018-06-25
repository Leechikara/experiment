# coding=utf-8
from data_utils import Vectorizer
from model import EmbeddingModel, EmbeddingAgent
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import *


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_task", type=str)
    parser.add_argument("--test_task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)

    test_file = os.path.join(DATA_ROOT, "public", args.test_task, "test.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", args.test_task + ".txt")

    vectorizer = Vectorizer()
    word2index, index2word, vocab_dim = vectorizer.load_vocab(args.trained_model)
    test_tensor = vectorizer.make_tensor(test_file, word2index)
    candidates_tensor = vectorizer.make_tensor(candidates_file, word2index)

    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.randomSeed, args.shareEmbedding)

    if args.trained_model is not None:
        with open(args.trained_model, "rb") as f:
            model.load_state_dict(torch.load(f))

    config = {"device": args.device}
    agent = EmbeddingAgent(config, model, None, None, test_tensor, candidates_tensor)
    agent.test()
