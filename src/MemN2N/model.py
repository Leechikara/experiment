# coding = utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from sklearn import metrics
import logging
import pickle
import os
import torch.nn.functional as F

sys.path.append("/home/wkwang/workstation/experiment/src")
from MemN2N.data_utils import batch_iter
from nn_utils.nn_utils import Attn, get_bow


class MemN2N(nn.Module):
    def __init__(self, config, candidates):
        super(MemN2N, self).__init__()
        torch.manual_seed(config["random_seed"])

        self.vocab_size = config["vocab_size"]
        self.emb_dim = config["emb_dim"]
        self.max_hops = config["max_hops"]
        self.attn_method = config["attn_method"]
        self.sent_emb_method = config["sent_emb_method"]
        self.emb_sum = config["emb_sum"]

        # A is embedding layer for context and query
        self.A = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # W is embedding layer for candidate responses
        self.W = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # Forward layer for hops
        self.H = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.attn_layer = Attn(self.attn_method, self.emb_dim, self.emb_dim)

        assert config["nonlinear"].lower() in ["tanh", "iden", "relu"]
        if config["nonlinear"].lower() == "tanh":
            self.nonlinear = nn.Tanh()
        elif config["nonlinear"].lower() == "iden":
            self.nonlinear = None
        elif config["nonlinear"].lower() == "relu":
            self.nonlinear == nn.ReLU()
        else:
            pass

        # register candidates for answer selecting
        self.register_buffer("candidates", torch.from_numpy(candidates))

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

    def forward(self, stories, queries):
        # Encode stories and queries
        if self.sent_emb_method == "bow":
            m, _ = get_bow(self.A(stories), self.emb_sum)
            q, _ = get_bow(self.A(queries), self.emb_sum)
        else:
            pass
        u = [q]

        for _ in range(self.max_hops):
            # attention over memory and read memory
            _, o_k = self.attn_layer(m, u[-1])

            # fuse read memory and previous hops
            u_k = self.H(u[-1]) + o_k
            if self.nonlinear is not None:
                u_k = self.nonlinear(u_k)

            u.append(u_k)

        if self.sent_emb_method == "bow":
            candidates_rep, _ = get_bow(self.W(self.candidates), self.emb_sum)
        logits = torch.mm(u[-1], candidates_rep.t())

        return logits


class MemAgent(object):
    def __init__(self, config, model, train_data, dev_data, test_data, data_utils):
        np.random.seed(config["random_seed"] + 1)
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.data_utils = data_utils
        self.logger = self._setup_logger()

    @staticmethod
    def _setup_logger():
        logging.basicConfig(
            format="[%(levelname)s] %(asctime)s: %(message)s (%(pathname)s:%(lineno)d)",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            stream=sys.stdout)
        logger = logging.getLogger("User-Agnostic-Dialog-MemoryNetWork")
        logger.setLevel(logging.DEBUG)
        return logger

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config["device"])

    def batch_fit(self, stories, queries, answers):
        self.optimizer.zero_grad()
        logits = self.model(stories, queries)
        loss = F.cross_entropy(logits, answers)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_clip"])
        self.optimizer.step()
        return loss.item()

    def batch_predict(self, stories, queries):
        preds = list()
        for stories_batch, queries_batch, _ in batch_iter(stories, queries, None, self.config["batch_size"], False):
            logits = self.model(self.tensor_wrapper(stories_batch), self.tensor_wrapper(queries_batch))
            predict_op = torch.argmax(logits, dim=1)
            pred = predict_op.detach().to(torch.device("cpu")).numpy()
            preds += list(pred)
        return preds

    def get_checkpoints(self, concerned_state=["A", "W", "H", "attn_layer"]):
        checkpoints = dict()
        for attr in concerned_state:
            checkpoints[attr] = self.model.__getattr__(attr)
        return checkpoints

    def train(self):
        self.logger.info("Run main with config {}".format(self.config))

        trainS, trainQ, trainA = self.data_utils.vectorize_data(self.train_data, self.config["batch_size"])
        valS, valQ, valA = self.data_utils.vectorize_data(self.dev_data, self.config["batch_size"])
        n_train = len(trainS)
        n_val = len(valS)
        print("Train Set Size", n_train)
        print("Dev Set Size", n_val)

        best_validation_accuracy = 0

        for epoch in range(1, self.config["epochs"] + 1):
            total_cost = 0.0
            for s, q, a in batch_iter(trainS, trainQ, trainA, batch_size=self.config["batch_size"], shuffle=True):
                with torch.set_grad_enabled(True):
                    cost_t = self.batch_fit(self.tensor_wrapper(s), self.tensor_wrapper(q), self.tensor_wrapper(a))
                total_cost += cost_t
            if epoch % self.config["evaluation_interval"] == 0:
                with torch.set_grad_enabled(False):
                    train_preds = self.batch_predict(trainS, trainQ)
                    val_preds = self.batch_predict(valS, valQ)
                train_acc = metrics.accuracy_score(np.array(train_preds), trainA)
                val_acc = metrics.accuracy_score(val_preds, valA)
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
                    with open(os.path.join(self.config["save_dir"], model_file), "wb") as f:
                        pickle.dump(self.get_checkpoints(), f)

    def test(self):
        testS, testQ, testA = self.data_utils.vectorize_data(self.test_data, self.config["batch_size"])
        n_test = len(testS)
        print("Testing Size", n_test)
        with torch.set_grad_enabled(False):
            test_preds = self.batch_predict(testS, testQ)
        test_acc = metrics.accuracy_score(test_preds, testA)
        print("Testing Accuracy:", test_acc)
