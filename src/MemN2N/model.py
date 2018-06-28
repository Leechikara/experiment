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

sys.path.append("/home/wkwang/workstation/experiment/src")
from MemN2N.data_utils import batch_iter


class MemN2N(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_hops, nonlinear, candidates, random_seed):
        super(MemN2N, self).__init__()
        torch.manual_seed(random_seed)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_hops = max_hops

        # A is embedding layer for context and query
        self.A = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # W is embedding layer for candidate responses
        self.W = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.H = nn.Linear(embedding_dim, embedding_dim, bias=False)

        assert nonlinear.lower() in ["tanh", "iden", "relu"]
        if nonlinear.lower() == "tanh":
            self.nonlinear = nn.Tanh()
        elif nonlinear.lower() == "iden":
            self.nonlinear = None
        elif nonlinear.lower() == "relu":
            self.nonlinear == nn.ReLU()
        else:
            pass

        # register candidates for answer selecting
        self.candidates = torch.from_numpy(candidates)
        self.register_buffer("candidates_const", self.candidates)

        self.softmax_layer = torch.nn.Softmax(dim=1)

    def load_checkpoints(self, f, mapping_dict={}):
        with open(f, 'rb') as f:
            checkpoints = pickle.load(f)

        def new_get_attr(_object_, attr_name):
            for attr_item in attr_name.split("."):
                _object_ = _object_.__getattr__(attr_item)
            return _object_

        def new_copy_parameter(source_p, target_p, mapping):
            if mapping is None:
                for idx in range(len(source_p)):
                    target_p[idx] = torch.from_numpy(source_p[idx])
            else:
                for source_idx, target_idx in mapping:
                    target_p[target_idx] = torch.from_numpy(source_p[source_idx])

        for key, value in checkpoints.items():
            target = new_get_attr(self, key)
            new_copy_parameter(value, target, mapping_dict.get(key, None))

    def forward(self, stories, queries):
        q_emb = self.A(queries)
        u_0 = torch.sum(q_emb, 1)
        u = [u_0]

        for _ in range(self.max_hops):
            m_emb = self.A(stories)
            m = torch.sum(m_emb, 2)
            u_temp = torch.transpose(torch.unsqueeze(u[-1], -1), 1, 2)
            dotted = torch.sum(m * u_temp, 2)

            # Calculate probabilities
            probs = self.softmax_layer(dotted)

            probs_temp = torch.transpose(torch.unsqueeze(probs, -1), 1, 2)
            c_temp = torch.transpose(m, 1, 2)
            o_k = torch.sum(c_temp * probs_temp, 2)

            u_k = self.H(u[-1]) + o_k
            if self.nonlinear is not None:
                u_k = self.nonlinear(u_k)

            u.append(u_k)

        candidate_emb = self.W(self.candidates_const)
        candidate_emb_sum = torch.sum(candidate_emb, 1)
        logits = torch.mm(u[-1], candidate_emb_sum.t())

        return logits


class MemAgent(object):
    def __init__(self, config, model, train_data, dev_data, test_data, data_utils):
        np.random.seed(config["random_seed"] + 1)
        self.config = config
        self.model = model.to(config["device"])
        self.loss = nn.CrossEntropyLoss()
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

    def _gradient_noise_and_clip(self, parameters):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        nn.utils.clip_grad_norm_(parameters, self.config["max_clip"])

        for p in parameters:
            noise = torch.randn(p.size()) * self.config["noise_stddev"]
            noise = noise.to(self.config["device"])
            p.grad.data.add_(noise)

    def batch_fit(self, stories, queries, answers):
        self.optimizer.zero_grad()
        logits = self.model(stories, queries)
        loss = self.loss(logits, answers)
        loss.backward()
        self._gradient_noise_and_clip(self.model.parameters())
        self.optimizer.step()
        return loss.item()

    def batch_predict(self, stories, queries):
        preds = list()
        for stories_batch, queries_batch, _ in batch_iter(stories, queries, None, self.config["batch_size"], False):
            logits = self.model(self.tensor_wrapper(stories_batch), self.tensor_wrapper(queries_batch))
            predict_op = torch.argmax(logits, dim=1)
            pred = predict_op.cpu().numpy()
            preds += list(pred)
        return preds

    def get_checkpoints(self, concerned_p=["A.weight", "W.weight", "H.weight"]):
        checkpoints = dict()
        for name, p in self.model.named_parameters():
            if name in concerned_p:
                checkpoints[name] = p.detach().to("cpu").numpy()
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
        test_preds = self.batch_predict(testS, testQ)
        test_acc = metrics.accuracy_score(test_preds, testA)
        print("Testing Accuracy:", test_acc)
