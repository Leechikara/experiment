# coding = utf-8
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT
from supervised_embedding.data_utils import Vectorizer
from supervised_embedding.model import EmbeddingModel, EmbeddingAgent


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negative_cand", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--shareEmbedding", action='store_true')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max_clip", type=float, default=40)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # checkpoints path
    args.save_dir = os.path.join(args.save_dir, args.task)

    train_file = os.path.join(DATA_ROOT, "public", args.task, "train.txt")
    dev_file = os.path.join(DATA_ROOT, "public", args.task, "dev.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", args.task + ".txt")

    vectorizer = Vectorizer()
    word2index, index2word, vocab_dim = vectorizer.load_vocab(args.task)
    train_tensor = vectorizer.make_tensor(train_file, word2index)
    dev_tensor = vectorizer.make_tensor(dev_file, word2index)
    candidates_tensor = vectorizer.make_tensor(candidates_file, word2index)

    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.random_seed, args.shareEmbedding)

    if args.trained_model is not None:
        if os.path.isdir(args.trained_model):
            trained_model = os.listdir(args.trained_model)
            trained_model.sort(key=lambda x: int(x.split("_")[1]))
            args.trained_model = os.path.join(args.trained_model, trained_model[-1])
        print("Using trained model in {}".format(args.trained_model))
        model.load_checkpoints(args.trained_model)

    config = {"lr": args.learning_rate, "epochs": args.epochs, "negative_cand": args.negative_cand,
              "device": args.device, "batch_size": args.batch_size, "save_dir": args.save_dir,
              "random_seed": args.random_seed, "max_clip": args.max_clip}
    agent = EmbeddingAgent(config, model, train_tensor, dev_tensor, None, candidates_tensor)
    agent.train()
