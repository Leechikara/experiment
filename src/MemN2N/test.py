# coding = utf-8
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from MemN2N.model import MemN2N, MemAgent
from config.config import DATA_ROOT
from MemN2N.data_utils import DataUtils, build_p_mapping


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--testing_task", type=str)
    parser.add_argument("--aware_new", action="store_true")
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--memory_size", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--nonlinear", type=str, default="iden")
    parser.add_argument("--attn_method", type=str, default="general")
    parser.add_argument("--sent_emb_method", type=str, default="bow")
    parser.add_argument("--emb_sum", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    test_file = os.path.join(DATA_ROOT, "public", args.testing_task, "test.txt")

    data_utils = DataUtils()
    # If the testing is aware of new domain
    if args.aware_new:
        data_utils.load_vocab(args.testing_task)
        data_utils.load_candidates(args.testing_task)

        # build mapping for parameters
        original_data_utils = DataUtils()
        original_data_utils.load_vocab(args.trained_task)
        mapping_dict = {"A": build_p_mapping(original_data_utils.word2index, data_utils.word2index),
                        "W": build_p_mapping(original_data_utils.word2index, data_utils.word2index)}
    else:
        data_utils.load_vocab(args.trained_task)
        data_utils.load_candidates(args.trained_task)
        mapping_dict = {}

    test_data = data_utils.load_dialog(test_file)
    data_utils.build_pad_config(test_data, args.memory_size)
    candidates = data_utils.vectorize_candidates()

    config = {"vocab_size": data_utils.vocab_size, "emb_dim": args.emb_dim, "max_hops": args.max_hops,
              "nonlinear": args.nonlinear, "random_seed": args.random_seed, "attn_method": args.attn_method,
              "sent_emb_method": args.sent_emb_method, "emb_sum": args.emb_sum}
    model = MemN2N(config, candidates).to(args.device)

    assert args.trained_model is not None
    if os.path.isdir(args.trained_model):
        trained_model = os.listdir(args.trained_model)
        trained_model.sort(key=lambda x: int(x.split("_")[1]))
        args.trained_model = os.path.join(args.trained_model, trained_model[-1])
    print("Using trained model in {}".format(args.trained_model))
    model.load_checkpoints(args.trained_model, mapping_dict)

    config = {"device": args.device, "batch_size": args.batch_size, "random_seed": args.random_seed}
    agent = MemAgent(config, model, None, None, test_data, data_utils)
    agent.test()
