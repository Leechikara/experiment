# coding = utf-8

import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from collections import defaultdict
import pickle
from sklearn import metrics

sys.path.append("/home/wkwang/workstation/experiment/src")
from Dual_rnn.data_apis.data_utils import batch_iter, DATA_ROOT, build_p_mapping
from nn_utils.nn_utils import rnn_seq, RnnV


class DualLSTM(nn.Module):
    def __init__(self, config, candidates):
        super(DualLSTM, self).__init__()
        torch.manual_seed(config.random_seed)

        self.vocab_size = config.word_emb_size
        self.rnn_hidden_size = config.rnn_hidden_size
        self.config = config

        self.ctx_encoder = RnnV(self.vocab_size, self.rnn_hidden_size, rnn_type="lstm", num_layers=config.rnn_layers,
                                bias=True, batch_first=True, dropout=config.rnn_dropout, bidirectional=config.rnn_bidirectional)
        self.response_encoder = RnnV(self.vocab_size, self.rnn_hidden_size, rnn_type="lstm",
                                     num_layers=config.rnn_layers,
                                     bias=True, batch_first=True, dropout=config.rnn_dropout,
                                     bidirectional=config.rnn_bidirectional)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=0)

        self.W = nn.Linear(self.config.sent_emb_size, self.config.sent_emb_size)

        # register candidates for answer selecting
        self.register_buffer("candidates", torch.from_numpy(candidates))

    def sent_encode(self, s):
        s_encode = rnn_seq(self.embedding(s), self.sent_rnn, self.config.sent_emb_size)
        return s_encode

    def load_checkpoints(self, f, mapping_dict={}):
        with open(f, 'rb') as f:
            checkpoints = pickle.load(f)

        for attr, source in checkpoints.items():
            mapping = mapping_dict.get(attr, None)
            target = self.__getattr__(attr)
            if mapping is None:
                for target_p, source_p in zip(target.parameters(), source.parameters()):
                    target_p.data = source_p.data
            else:
                for source_idx, target_idx in mapping:
                    target.weight.data[target_idx] = source.weight.data[source_idx]

    def forward(self, ctx):
        ctx_emb = rnn_seq(self.embedding(ctx), self.ctx_encoder, self.config.sent_emb_size)
        response_emb = rnn_seq(self.embedding(self.candidates), self.response_encoder, self.config.sent_emb_size)
        logits = torch.mm(self.W(ctx_emb), response_emb.t())
        return logits

    def load_checkpoints(self, f, mapping_dict={}):
        with open(f, 'rb') as f:
            checkpoints = pickle.load(f)

        for attr, source in checkpoints.items():
            mapping = mapping_dict.get(attr, None)
            target = self.__getattr__(attr)
            if mapping is None:
                for target_p, source_p in zip(target.parameters(), source.parameters()):
                    target_p.data = source_p.data
            else:
                for source_idx, target_idx in mapping:
                    target.weight.data[target_idx] = source.weight.data[source_idx]


class DualLSTMAgent(object):
    def __init__(self, config, model, train_data, dev_data, test_data, data_utils):
        np.random.seed(config.random_seed + 1)
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.data_utils = data_utils
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config.device)

    def batch_fit(self, ctx, resp):
        self.optimizer.zero_grad()
        logits = self.model(ctx)
        loss = F.cross_entropy(logits, resp)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_clip)
        self.optimizer.step()
        return loss.item()

    def batch_predict(self, ctx):
        preds = list()
        for ctx, _ in batch_iter(ctx, None, self.config.batch_size, False):
            logits = self.model(self.tensor_wrapper(ctx))
            predict_op = torch.argmax(logits, dim=1)
            pred = predict_op.detach().to(torch.device("cpu")).numpy()
            preds += list(pred)
        return preds

    def get_checkpoints(self, concerned_state=["ctx_encoder", "response_encoder", "embedding", "W"]):
        checkpoints = dict()
        for attr in concerned_state:
            checkpoints[attr] = self.model.__getattr__(attr)
        return checkpoints

    def train(self):
        train_ctx, train_resp = self.data_utils.vectorize_data(self.train_data, self.config.batch_size)
        val_ctx, val_resp = self.data_utils.vectorize_data(self.dev_data, self.config.batch_size)
        n_train = len(train_ctx)
        n_val = len(val_ctx)
        print("Train Set Size", n_train)
        print("Dev Set Size", n_val)

        best_validation_accuracy = 0

        for epoch in range(1, self.config.epochs + 1):
            total_cost = 0.0
            for ctx, resp in batch_iter(train_ctx, train_resp, batch_size=self.config.batch_size, shuffle=True):
                with torch.set_grad_enabled(True):
                    cost_t = self.batch_fit(self.tensor_wrapper(ctx), self.tensor_wrapper(resp))
                total_cost += cost_t
            if epoch % self.config.evaluation_interval == 0:
                with torch.set_grad_enabled(False):
                    train_preds = self.batch_predict(train_ctx)
                    val_preds = self.batch_predict(val_ctx)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_resp)
                val_acc = metrics.accuracy_score(val_preds, val_resp)
                print('-----------------------')
                print('Epoch', epoch)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    model_file = "epoch_{}_accuracy_{}.pkl".format(epoch, val_acc)
                    # A different method to store parameters
                    with open(os.path.join(self.config.save_dir, model_file), "wb") as f:
                        pickle.dump(self.get_checkpoints(), f)

    def test(self):
        test_ctx, test_resp = self.data_utils.vectorize_data(self.test_data, self.config.batch_size)
        n_test = len(test_ctx)
        print("Testing Size", n_test)
        with torch.set_grad_enabled(False):
            test_preds = self.batch_predict(test_ctx)
        test_acc = metrics.accuracy_score(test_preds, test_resp)
        print("Testing Accuracy:", test_acc)
