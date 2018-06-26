# coding = utf-8
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from MemN2N.model import MemN2N, MemAgent
from config.config import DATA_ROOT
from MemN2N.data_utils import DataUtils


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--test_task", type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--memory_size", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--nonlinear", type=str, default="iden")
    parser.add_argument("--max_clip", type=float, default=40)
    parser.add_argument("--noise_stddev", type=float, default=0.1)
    parser.add_argument("--evaluation_interval", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # checkpoints path
    args.save_dir = "/".join([args.save_dir, args.task])

    train_file = os.path.join(DATA_ROOT, "public", args.task, "train.txt")
    dev_file = os.path.join(DATA_ROOT, "public", args.task, "dev.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", args.task + ".txt")

    data_utils = DataUtils()
    data_utils.load_vocab(args.task)
    data_utils.load_candidates(args.task)
    train_data = data_utils.load_dialog(train_file)
    dev_data = data_utils.load_dialog(dev_file)
    data = train_data + dev_data
    data_utils.build_pad_config(data, args.memory_size)
    candidates_vec = data_utils.vectorize_candidates()

    model = MemN2N(data_utils.vocab_size, args.emb_dim, args.max_hops, args.nonlinear, candidates_vec,
                   args.random_seed).to(args.device)

    if args.trained_model is not None:
        with open(args.trained_model, "rb") as f:
            model.load_state_dict(torch.load(f))

    config = {"lr": args.learning_rate, "epochs": args.epochs, "device": args.device, "batch_size": args.batch_size,
              "save_dir": args.save_dir, "max_clip": args.max_clip, "noise_stddev": args.noise_stddev,
              "random_seed": args.random_seed, "evaluation_interval": args.evaluation_interval}
    agent = MemAgent(config, model, train_data, dev_data, None, data_utils)
    agent.train()
