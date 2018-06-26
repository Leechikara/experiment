# coding = utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import logging
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from supervised_embedding.data_utils import batch_iter, neg_sampling_iter


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_dim, emb_dim, margin, random_seed=42, share_parameter=False):
        super(EmbeddingModel, self).__init__()
        torch.manual_seed(random_seed)

        self._vocab_dim = vocab_dim
        self._emb_dim = emb_dim
        self._margin = margin
        self.share_parameter = share_parameter

        self.context_embedding = Parameter(torch.Tensor(emb_dim, vocab_dim))
        self.response_embedding = Parameter(torch.Tensor(emb_dim, vocab_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.context_embedding.data.uniform_(-1, 1)
        if self.share_parameter:
            self.response_embedding = self.context_embedding
        else:
            self.response_embedding.data.uniform_(-1, 1)

    def forward(self, context_batch, response_batch, neg_response_batch):
        cont_rep = torch.t(torch.mm(self.context_embedding, torch.t(context_batch)))
        resp_rep = torch.mm(self.response_embedding, torch.t(response_batch))
        neg_resp_rep = torch.mm(self.response_embedding, torch.t(neg_response_batch))

        f_pos = torch.diag(torch.mm(cont_rep, resp_rep))
        f_neg = torch.diag(torch.mm(cont_rep, neg_resp_rep))

        loss = torch.sum(F.relu(f_neg - f_pos + self._margin))
        return f_pos, f_neg, loss


class EmbeddingAgent(object):
    def __init__(self, config, model, train_tensor, dev_tensor, test_tensor, candidates_tensor):
        np.random.seed(config["random_seed"] + 1)
        self.config = config
        self.model = model
        self.train_tensor = train_tensor
        self.dev_tensor = dev_tensor
        self.test_tensor = test_tensor
        self.candidates_tensor = candidates_tensor
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        self.logger = self._setup_logger()

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.asarray(data)
        data = torch.from_numpy(data)
        return data.to(self.config["device"])

    def _train(self, batch_size, neg_size):
        avg_loss = 0
        for batch in batch_iter(self.train_tensor, batch_size, True):
            for neg_batch in neg_sampling_iter(self.train_tensor, batch_size, neg_size):
                self.optimizer.zero_grad()
                _, _, loss = self.model(self.tensor_wrapper(batch[:, 0, :]),
                                        self.tensor_wrapper(batch[:, 1, :]),
                                        self.tensor_wrapper(neg_batch[:, 1, :]))
                loss.backward()

                # clamp grad
                for param in self.model.parameters():
                    param.grad.data.clamp_(-5, 5)

                self.optimizer.step()
                avg_loss += loss.item()
        avg_loss = avg_loss / (self.train_tensor.shape[0] * neg_size)
        return avg_loss

    def _forward_all(self, batch_size):
        avg_dev_loss = 0
        for batch in batch_iter(self.dev_tensor, batch_size):
            for neg_batch in neg_sampling_iter(self.dev_tensor, batch_size, count=1, seed=42):
                _, _, loss = self.model(self.tensor_wrapper(batch[:, 0, :]),
                                        self.tensor_wrapper(batch[:, 1, :]),
                                        self.tensor_wrapper(neg_batch[:, 1, :]))
                avg_dev_loss += loss.item()
        avg_dev_loss = avg_dev_loss / self.dev_tensor.shape[0]
        return avg_dev_loss

    @staticmethod
    def _setup_logger():
        logging.basicConfig(
            format="[%(levelname)s] %(asctime)s: %(message)s (%(pathname)s:%(lineno)d)",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            stream=sys.stdout)
        logger = logging.getLogger("User-Agnostic-Dialog-EmbeddingModel")
        logger.setLevel(logging.DEBUG)
        return logger

    def evaluate_one_row(self, candidates_tensor, true_context, test_score, true_response, batch_size):
        for batch in batch_iter(candidates_tensor, batch_size):
            candidate_responses = batch[:, 0, :]
            context_batch = np.repeat(true_context, candidate_responses.shape[0], axis=0)

            scores, _, _ = self.model(self.tensor_wrapper(context_batch),
                                      self.tensor_wrapper(candidate_responses),
                                      self.tensor_wrapper(candidate_responses))
            scores = scores.detach().cpu().numpy()

            for ind, score in enumerate(scores):
                if score == float("Inf") or score == -float("Inf") or score == float("NaN"):
                    print(score, ind, scores[ind])
                    raise ValueError
                if score >= test_score and not np.array_equal(candidate_responses[ind], true_response):
                    return False
        return True

    def evaluate(self, test_tensor, candidates_tensor, batch_size):
        neg = 0
        pos = 0
        for row in test_tensor:
            true_context = [row[0]]
            test_score, _, _ = self.model(self.tensor_wrapper(true_context),
                                          self.tensor_wrapper([row[1]]),
                                          self.tensor_wrapper([row[1]]))
            test_score = test_score.item()

            is_pos = self.evaluate_one_row(candidates_tensor, true_context, test_score, row[1], batch_size)
            if is_pos:
                pos += 1
            else:
                neg += 1
        return pos, neg, pos / (pos + neg)

    def test(self):
        pos, neg, rate = self.evaluate(self.test_tensor, self.candidates_tensor, self.config["batch_size"])
        print("pos:{} neg:{} rate:{}".format(pos, neg, rate))

    def train(self):
        self.logger.info("Run main with config {}".format(self.config))

        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        negative_cand = self.config["negative_cand"]
        save_dir = self.config["save_dir"]

        prev_best_accuracy = 0

        for epoch in range(epochs):
            with torch.set_grad_enabled(True):
                avg_loss = self._train(batch_size, negative_cand)
            with torch.set_grad_enabled(False):
                avg_dev_loss = self._forward_all(batch_size)

            self.logger.info("Epoch: {}; Train loss: {}; Dev loss: {};".format(epoch, avg_loss, avg_dev_loss))

            if epoch % 2 == 0:
                with torch.set_grad_enabled(False):
                    dev_eval = self.evaluate(self.dev_tensor, self.candidates_tensor, batch_size)
                self.logger.info("Evaluation: {}".format(dev_eval))
                accuracy = dev_eval[2]
                if accuracy >= prev_best_accuracy:
                    self.logger.debug("Saving checkpoint")
                    prev_best_accuracy = accuracy
                    model_file = "epoch_{}_accuracy_{}.pkl".format(epoch, accuracy)
                    torch.save(self.model.state_dict(), save_dir + "/" + model_file)
