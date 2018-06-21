# coding=utf-8
from make_tensor import Vectorizer
from model import EmbeddingModel, EmbeddingAgent
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import *


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str)
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

    task = args.task

    test_file = os.path.join(DATA_ROOT, "public", task, "test.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", task + ".txt")

    vectorizer = Vectorizer(task)
    test_tensor = vectorizer.make_tensor(test_file)
    candidates_tensor = vectorizer.make_tensor(candidates_file)

    vocab_dim = vectorizer.vocab_dim()
    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.randomSeed, args.shareEmbedding)

    if args.trained_model is not None:
        with open(args.trained_model, "rb") as f:
            model.load_state_dict(torch.load(f))

    config = {"device": args.device}
    agent = EmbeddingAgent(config, model, None, None, test_tensor, candidates_tensor)
    agent.test()
