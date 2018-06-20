# coding = utf-8
from make_tensor import Vectorizer
from model import EmbeddingModel, EmbeddingAgent
import argparse
import torch
import os
from config import *


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--negative_cand", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--shareEmbedding", action='store_true')
    parser.add_argument("--randomSeed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)

    task = args.task
    train_file = os.path.join(DATA_ROOT, "public", task, "train.txt")
    dev_file = os.path.join(DATA_ROOT, "public", task, "dev.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", task + ".txt")

    vectorizer = Vectorizer(task)
    train_tensor = vectorizer.make_tensor(train_file)
    dev_tensor = vectorizer.make_tensor(dev_file)
    candidates_tensor = vectorizer.make_tensor(candidates_file)

    vocab_dim = vectorizer.vocab_dim()
    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.randomSeed, args.shareEmbedding)

    if args.trained_model is not None:
        with open(args.trained_model, "rb") as f:
            model.load_state_dict(torch.load(f))

    config = {"lr": args.learning_rate, "epochs": args.epochs, "negative_cand": args.negative_cand,
              "device": args.device, "batch_size": args.batch_size, "save_dir": args.save_dir}
    agent = EmbeddingAgent(config, model, train_tensor, dev_tensor, None, candidates_tensor)
    agent.train()
