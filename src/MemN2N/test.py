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
    parser.add_argument("--testing_task", type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--memory_size", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--nonlinear", type=str, default="iden")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    test_file = os.path.join(DATA_ROOT, "public", args.testing_task, "test.txt")
    candidates_file = os.path.join(DATA_ROOT, "candidate", args.trained_task + ".txt")

    data_utils = DataUtils()
    data_utils.load_vocab(args.trained_task)
    data_utils.load_candidates(args.trained_task)
    test_data = data_utils.load_dialog(test_file)
    data_utils.build_pad_config(test_data, args.memory_size)
    candidates_vec = data_utils.vectorize_candidates()

    model = MemN2N(data_utils.vocab_size, args.emb_dim, args.max_hops, args.nonlinear, candidates_vec,
                   args.random_seed).to(args.device)

    assert args.trained_model is not None
    with open(args.trained_model, "rb") as f:
        model.load_state_dict(torch.load(f))

    config = {"device": args.device, "batch_size": args.batch_size}
    agent = MemAgent(config, model, None, None, test_data, data_utils)
    agent.test()
