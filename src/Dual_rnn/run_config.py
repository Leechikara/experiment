# coding = utf-8
import torch


class RunConfig(object):
    system_mode = "train"
    trained_task = "task_1"
    save_dir = "checkpoints/task_1"
    trained_model = None

    aware_new = False
    testing_task = "task_5"

    random_seed = 42
    lr = 0.001
    batch_size = 32
    max_clip = 40.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 60
    evaluation_interval = 2

    word_emb_size = 32

    rnn_type = "lstm"
    rnn_hidden_size = 32
    rnn_layers = 1
    rnn_dropout = 0
    rnn_bidirectional = True
    ctx_window = 4

    sent_emb_size = rnn_hidden_size * rnn_layers * 2 if rnn_bidirectional \
        else rnn_hidden_size * rnn_layers