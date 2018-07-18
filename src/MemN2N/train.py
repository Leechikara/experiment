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

    parser.add_argument("--task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--memory_size", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--nonlinear", type=str, default="iden")
    parser.add_argument("--max_clip", type=float, default=40)
    parser.add_argument("--attn_method", type=str, default="general")
    parser.add_argument("--sent_emb_method", type=str, default="bow")
    parser.add_argument("--emb_sum", action="store_true")
    parser.add_argument("--evaluation_interval", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # checkpoints path
    args.save_dir = os.path.join(args.save_dir, args.task)

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
    candidates = data_utils.vectorize_candidates()

    config = {"vocab_size": data_utils.vocab_size, "emb_dim": args.emb_dim, "max_hops": args.max_hops,
              "nonlinear": args.nonlinear, "random_seed": args.random_seed, "attn_method": args.attn_method,
              "sent_emb_method": args.sent_emb_method, "emb_sum": args.emb_sum}
    model = MemN2N(config, candidates).to(args.device)

    if args.trained_model is not None:
        if os.path.isdir(args.trained_model):
            trained_model = os.listdir(args.trained_model)
            trained_model.sort(key=lambda x: int(x.split("_")[1]))
            args.trained_model = os.path.join(args.trained_model, trained_model[-1])
        print("Using trained model in {}".format(args.trained_model))
        model.load_checkpoints(args.trained_model)

    config = {"lr": args.learning_rate, "epochs": args.epochs, "device": args.device, "batch_size": args.batch_size,
              "save_dir": args.save_dir, "max_clip": args.max_clip, "random_seed": args.random_seed,
              "evaluation_interval": args.evaluation_interval}
    agent = MemAgent(config, model, train_data, dev_data, None, data_utils)
    agent.train()
