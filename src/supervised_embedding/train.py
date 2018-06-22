# coding = utf-8
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
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negative_cand", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shareEmbedding", action='store_true')
    parser.add_argument("--randomSeed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # checkpoints path
    args.save_dir = "/".join([args.save_dir, args.task])

    task = args.task
    train_file = os.path.join(DATA_ROOT, "public", task, "train.txt")
    dev_file = os.path.join(DATA_ROOT, "public", task, "dev.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", task + ".txt")

    vectorizer = Vectorizer(task)
    train_tensor = vectorizer.make_tensor(train_file)
    dev_tensor = vectorizer.make_tensor(dev_file)
    candidates_tensor = vectorizer.make_tensor(candidates_file)

    vocab_dim = vectorizer.vocab_dim()
    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.randomSeed, args.shareEmbedding).to(args.device)

    if args.trained_model is not None:
        with open(args.trained_model, "rb") as f:
            model.load_state_dict(torch.load(f))

    config = {"lr": args.learning_rate, "epochs": args.epochs, "negative_cand": args.negative_cand,
              "device": args.device, "batch_size": args.batch_size, "save_dir": args.save_dir}
    agent = EmbeddingAgent(config, model, train_tensor, dev_tensor, None, candidates_tensor)
    agent.train()
