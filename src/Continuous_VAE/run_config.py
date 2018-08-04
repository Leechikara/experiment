# coding = utf-8
import torch


class RunConfig(object):
    trained_model = None
    coming_task = "task_1"
    system_mode = "deploy"
    model_save_path = "checkpoints/model_H_RNN_task_1_deploy.pkl"
    debug_path = "debug/loss_H_RNN_task_1_deploy.pkl"

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
    ctx_encode_method = "HierarchalRNN"
    attn_method = "general"
    memory_size = 50
    max_hops = 3
    memory_nonlinear = "iden"
    ctx_rnn_type = "gru"
    ctx_rnn_hidden_size = 32
    ctx_rnn_layers = 1
    ctx_rnn_dropout = 0
    ctx_rnn_bidirectional = False

    sent_self_attn = True
    sent_self_attn_hidden = 32
    sent_self_attn_head = 1

    ctx_self_attn = False
    ctx_self_attn_hidden = 32
    ctx_self_attn_head = 1

    latent_size = 20
    prior_sample = 50
    posterior_sample = 50
    threshold = 0.7
    full_kl_step = 10000

    cluster_loss_available = True
    max_clusters = 8
    max_samples = 64
    cluster_loss_factor = 1

    if sent_encode_method == "rnn":
        if sent_self_attn is True:
            sent_emb_size = sent_rnn_hidden_size * 2 if sent_rnn_bidirectional else sent_rnn_hidden_size
        else:
            sent_emb_size = sent_rnn_hidden_size * sent_rnn_layers * 2 if sent_rnn_bidirectional \
                else sent_rnn_hidden_size * sent_rnn_layers
    elif sent_encode_method == "bow":
        sent_emb_size = word_emb_size

    if ctx_encode_method == "MemoryNetwork" or ctx_encode_method == "HierarchalSelfAttn":
        ctx_emb_size = sent_emb_size
    elif ctx_encode_method == "HierarchalRNN":
        if ctx_self_attn is True:
            ctx_emb_size = ctx_rnn_hidden_size * 2 if ctx_rnn_bidirectional else ctx_rnn_hidden_size
        else:
            ctx_emb_size = ctx_rnn_hidden_size * ctx_rnn_layers * 2 if ctx_rnn_bidirectional \
                else ctx_rnn_hidden_size * ctx_rnn_layers
