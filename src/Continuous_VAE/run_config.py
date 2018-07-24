# coding = utf-8
import torch


class RunConfig(object):
    trained_model = None
    save_dir = "checkpoints"

    random_seed = 42
    lr = 0.001
    batch_size = 32
    max_clip = 40.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # word embedding config
    word_emb_size = 32

    # sentence encoding config
    sent_encode_method = "rnn"
    emb_sum = False
    sent_rnn_type = "gru"
    sent_rnn_hidden_size = 32
    sent_rnn_layers = 1
    sent_rnn_dropout = 0
    sent_rnn_bidirectional = True

    # context encoding config
    ctx_encode_method = "MemoryNetwork"
    attn_method = "general"
    memory_size = 50
    max_hops = 3
    memory_nonlinear = "iden"
    ctx_rnn_type = "gru"
    ctx_rnn_hidden_size = 32
    ctx_rnn_layers = 1
    ctx_rnn_dropout = 0
    ctx_rnn_bidirectional = True

    self_attn = True
    self_attn_hidden = 32
    self_attn_head = 1

    latent_size = 20
    sample = 50
    threshold = 0.7

    if sent_encode_method == "rnn":
        if self_attn is True:
            sent_emb_size = sent_rnn_hidden_size * 2 if sent_rnn_bidirectional else sent_rnn_hidden_size
        else:
            sent_emb_size = sent_rnn_hidden_size * sent_rnn_layers * 2 if sent_rnn_bidirectional \
                else sent_rnn_hidden_size * sent_rnn_layers
    elif sent_encode_method == "bow":
        sent_emb_size = word_emb_size

    if ctx_encode_method == "MemoryNetwork":
        ctx_emb_size = sent_emb_size
    elif ctx_encode_method == "HierarchalRNN" or ctx_encode_method == "RNN":
        if self_attn is True:
            ctx_emb_size = ctx_rnn_hidden_size * 2 if ctx_rnn_bidirectional else ctx_rnn_hidden_size
        else:
            ctx_emb_size = ctx_rnn_hidden_size * ctx_rnn_layers * 2 if ctx_rnn_bidirectional \
                else ctx_rnn_hidden_size * ctx_rnn_layers
