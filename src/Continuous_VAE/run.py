# coding = utf-8
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from Continuous_VAE.model.cvae import ContinuousAgent, ContinuousVAE
from Continuous_VAE.data_apis.data_utils import DataUtils


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max_clip", type=float, default=40)
    parser.add_argument("--memory_size", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--nonlinear", type=str, default="iden")
    parser.add_argument("--context_emb_method", type=str, default="MemoryNetwork")
    parser.add_argument("--sent_emb_method", type=str, default="bow")
    parser.add_argument("--memory_nonlinear", type=str, default="iden")
    parser.add_argument("--attn_method", type=str, default="general")
    parser.add_argument("--emb_sum", action="store_true")
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--evaluation_interval", type=int, default=2)
    parser.add_argument("--save_dir", default="checkpoints", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    api = DataUtils()
    api.load_vocab()
    api.load_candidates()
    api.load_dialog()
    api.build_pad_config(args.memory_size)

    config = {"random_seed": args.random_seed, "emb_dim": args.emb_dim,
              "context_emb_method": args.context_emb_method, "memory_nonlinear": args.memory_nonlinear,
              "latent_size": args.latent_size, "max_hops": args.max_hops, "sample": args.sample,
              "threshold": args.threshold, "attn_method": args.attn_method, "emb_sum": args.emb_sum,
              "sent_emb_method": args.sent_emb_method}
    model = ContinuousVAE(config, api)

    if args.trained_model is not None:
        if os.path.isdir(args.trained_model):
            trained_model = os.listdir(args.trained_model)
            trained_model.sort(key=lambda x: int(x.split("_")[1]))
            args.trained_model = os.path.join(args.trained_model, trained_model[-1])
        print("Using trained model in {}".format(args.trained_model))
        model.load_state_dict(torch.load(args.trained_model))

    config = {"random_seed": args.random_seed, "device": args.device, "lr": args.learning_rate,
              "batch_size": args.batch_size, "max_clip": args.max_clip, "save_dir": args.save_dir}
    agent = ContinuousAgent(config, model, api)
    agent.simulate_run()
