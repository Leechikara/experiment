# coding = utf-8
import torch


class RunConfig(object):
    trained_model = None
    save_dir = "checkpoints"

    random_seed = 42
    learning_rate = 0.001
    batch_size = 32
    max_clip = 40.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # context encoding config
    context_encode_method = "MemoryNetwork"
    memory_size = 50
    max_hops = 3
    memory_nonlinear = "iden"

    # sentence encoding config
    sent_encode_method = "bow"
    emb_sum = False
    rnn_type = "gru"
    rnn_hidden_size = 32
    rnn_layers = 1
    rnn_dropout = 0
    bidirectional = True

    self_att = False
    attn_method = "general"

    latent_size = 20
    sample = 50
    threshold = 0.7
