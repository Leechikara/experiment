# coding=utf-8
import argparse
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from config.config import DATA_ROOT
from supervised_embedding.data_utils import Vectorizer, build_p_mapping
from supervised_embedding.model import EmbeddingModel, EmbeddingAgent


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_task", type=str)
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--testing_task", type=str)
    parser.add_argument("--aware_new", action="store_true")
    parser.add_argument("--emb_dim", default=32, type=int)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--shareEmbedding", action='store_true')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    test_file = os.path.join(DATA_ROOT, "public", args.testing_task, "test.txt")
    vectorizer = Vectorizer()

    # If the testing is aware of new domain
    if args.aware_new:
        candidates_file = os.path.join(DATA_ROOT, "candidate", args.testing_task + ".txt")
        word2index, index2word, vocab_dim = vectorizer.load_vocab(args.testing_task)
        original_word2index, _, _ = vectorizer.load_vocab(args.trained_task)
        mapping_dict = {"context_embedding": build_p_mapping(original_word2index, word2index),
                        "response_embedding": build_p_mapping(original_word2index, word2index)}
    else:
        candidates_file = os.path.join(DATA_ROOT, "candidate", args.trained_task + ".txt")
        word2index, index2word, vocab_dim = vectorizer.load_vocab(args.trained_task)
        mapping_dict = {}

    test_tensor = vectorizer.make_tensor(test_file, word2index)
    candidates_tensor = vectorizer.make_tensor(candidates_file, word2index)

    model = EmbeddingModel(vocab_dim, args.emb_dim, args.margin, args.random_seed, args.shareEmbedding)

    assert args.trained_model is not None
    if os.path.isdir(args.trained_model):
        trained_model = os.listdir(args.trained_model)
        trained_model.sort(key=lambda x: int(x.split("_")[1]))
        args.trained_model = os.path.join(args.trained_model, trained_model[-1])
    print("Using trained model in {}".format(args.trained_model))
    model.load_checkpoints(args.trained_model, mapping_dict)

    config = {"device": args.device, "batch_size": args.batch_size, "random_seed": args.random_seed}
    agent = EmbeddingAgent(config, model, None, None, test_tensor, candidates_tensor)
    agent.test()
