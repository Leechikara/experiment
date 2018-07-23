# coding = utf-8

import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import logging
from collections import defaultdict
import pickle
from sklearn import metrics

sys.path.append("/home/wkwang/workstation/experiment/src")
from Continuous_VAE.model.utils import sample_gaussian, gaussian_kld
from Continuous_VAE.data_apis.data_utils import batch_iter, TASKS, DATA_ROOT
from nn_utils.nn_utils import Attn, bow_sentence


class ContinuousVAE(nn.Module):
    def __init__(self, config, api):
        super(ContinuousVAE, self).__init__()
        torch.manual_seed(config["random_seed"])

        self.api = api
        self.config = config
        self.vocab_size = api.vocab_size
        self.emb_dim = config["emb_dim"]
        self.attn_method = config["attn_method"]
        self.sent_emb_method = config["sent_emb_method"]

        # Embedding for context and response
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # todo: Encoding method for context and response, Now we only implement the basic method
        if config["context_emb_method"] == "MemoryNetwork":
            self.attn_layer = Attn(self.attn_method, self.emb_dim, self.emb_dim)
            self.hops_map = nn.Linear(self.emb_dim, self.emb_dim)
            assert config["memory_nonlinear"].lower() in ["tanh", "iden", "relu"]
            if config["memory_nonlinear"].lower() == "tanh":
                self.memory_nonlinear = nn.Tanh()
            elif config["memory_nonlinear"].lower() == "relu":
                self.memory_nonlinear == nn.ReLU()
            else:
                self.memory_nonlinear = None

        cond_embedding_size = self.emb_dim
        response_embedding_size = self.emb_dim
        recog_input_size = cond_embedding_size + response_embedding_size

        # todo: make prior and posterior more expressive (IAF)
        # recognitionNetwork: A MLP
        self.recogNet_mulogvar = nn.Sequential(
            nn.Linear(recog_input_size, max(50, config["latent_size"] * 2)),
            nn.Tanh(),
            nn.Linear(max(50, config["latent_size"] * 2), config["latent_size"] * 2)
        )

        # priorNetwork: A MLP
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_embedding_size, max(50, config["latent_size"] * 2)),
            nn.Tanh(),
            nn.Linear(max(50, config["latent_size"] * 2), config["latent_size"] * 2)
        )

        # record all candidates in advance and current available index
        # If new candidates are known, we add its index into  available_cand_index
        # I think it's a more efficient strategy to simulate continuous leaning
        # In the beginning, only the candidates response in task_1 is available
        self.available_cand_index = list()
        with open(os.path.join(DATA_ROOT, "candidate", "task_1.txt")) as f:
            for line in f:
                line = line.strip()
                self.available_cand_index.append(api.candid2index[line])
        self.available_cand_index.sort()
        self.register_buffer("candidates", torch.from_numpy(api.vectorize_candidates()))

        # fuse cond_embedding and z
        # todo: there may be other method
        self.fused_cond_z = nn.Linear(self.emb_dim + config["latent_size"], self.emb_dim)

    def context_encoding_m2n(self, contexts):
        # encoding the contexts in a batch by memory networks
        # todo: more complex method for sentence embedding
        stories, queries = contexts
        if self.sent_emb_method == "bow":
            m, _ = bow_sentence(self.embedding(stories), self.config["emb_sum"])
            q, _ = bow_sentence(self.embedding(queries), self.config["emb_sum"])
        else:
            pass
        u = [q]

        for _ in range(self.config["max_hops"]):
            # attention over memory and read memory
            _, o_k = self.attn_layer(m, u[-1])

            # fuse read memory and previous hops
            u_k = self.hops_map(u[-1]) + o_k
            if self.memory_nonlinear is not None:
                u_k = self.memory_nonlinear(u_k)

            u.append(u_k)

        return u[-1]

    def threshold_method(self, sampled_response):
        uncertain_index = list()
        certain_index = list()
        certain_response = list()

        # todo: can we accelerate this part in GPU?
        for i, response_dist in enumerate(sampled_response):
            vot_result, vot_num = Counter(response_dist).most_common(1)[0]
            if vot_num < self.config["threshold"] * self.config["sample"]:
                uncertain_index.append(i)
            else:
                certain_index.append(i)
                certain_response.append(self.available_cand_index[vot_result])

        return uncertain_index, certain_index, certain_response

    def select_uncertain_points(self, logits):
        probs = F.softmax(logits, 2)
        probs = probs.contiguous().view(-1, len(self.available_cand_index))
        sampled_response = torch.multinomial(probs, 1)
        sampled_response = sampled_response.view(-1, self.config["sample"])
        sampled_response = sampled_response.detach().data.cpu().numpy()

        # todo: more heuristic method for uncertain points
        uncertain_index, certain_index, certain_response = self.threshold_method(sampled_response)

        return uncertain_index, certain_index, certain_response

    @staticmethod
    def evaluate(certain_indexs, certain_responses, feed_responses):
        feed_responses = np.array([feed_responses[i] for i in certain_indexs])
        certain_responses = np.array(certain_responses)
        acc = metrics.accuracy_score(feed_responses, certain_responses)
        return acc

    def helper(self, s):
        s = s.data.cpu().numpy()
        if s.ndim == 2:
            for l in s:
                l = list(filter(lambda x: x != 0, l))
                l = " ".join([self.api.index2word[x] for x in l])
                print(l)
        else:
            l = list(filter(lambda x: x != 0, s))
            l = " ".join([self.api.index2word[x] for x in l])
            print(l)

    def forward(self, feed_dict):
        """
        Step1: Get the context representation
        Step2: Sample z from prior
        Step3: Get the slice of uncertain points and certain points
        Step4: Calculate the IR Evaluation on the certain points and uncertain points
        Step5: Get the loss
        """
        # Step1: Get the context representation
        # todo: Now the context encoding is from Memory network and very naive
        context_rep = self.context_encoding_m2n(feed_dict["contexts"])

        # todo: we may add more to cond_embedding
        cond_embedding = context_rep

        # Step2: Sample z from prior
        prior_mulogvar = self.priorNet_mulogvar(cond_embedding)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
        latent_prior = sample_gaussian(prior_mu, prior_logvar, self.config["sample"])

        # Step3: Get the slice of uncertain points and certain points
        #    step1: fusing z and cond_embedding
        #    step2: Get the embedding of current candidates
        #    step3: Get the logits of each candidate
        #    step4: A method to judge which context is uncertain

        # step1: fusing z and cond_embedding
        # todo: The z can be seen as a gate or other complex method
        # todo: Now we do it very naive
        cond_embedding_temp = cond_embedding.unsqueeze(1).expand(-1, self.config["sample"], -1)
        cond_z_embed_prior = self.fused_cond_z(torch.cat([cond_embedding_temp, latent_prior], 2))
        # step2: Get the embedding of current candidates
        # todo: more complicated methods for candidates representations
        if self.sent_emb_method == "bow":
            candidates_rep, _ = bow_sentence(self.embedding(self.candidates), self.config["emb_sum"])
        else:
            pass
        current_candidates_rep = candidates_rep[self.available_cand_index]
        # step3: Get the logits of each candidate
        # todo: more complicated methods for candidates scoring
        logits = torch.matmul(cond_z_embed_prior, current_candidates_rep.t())
        # step4: A method to judge which context is uncertain
        uncertain_index, certain_index, certain_response = self.select_uncertain_points(logits)

        # Step4: Calculate the IR Evaluation on the certain points and uncertain points
        # todo: we left this part aside
        if len(certain_index) > 0:
            acc = self.evaluate(certain_index, certain_response, feed_dict["responses"])
        else:
            acc = 0

        # Step5: Get the loss
        #    step1: Simulate human in the loop and update the available response set
        #    step2: Get the uncertain cond & resp embedding
        #    step3: fuse cond & resp
        #    step4: Get z posterior
        #    step5: fuse z and cond

        if len(uncertain_index) > 0:
            # step1: Simulate human in the loop and update the available response set
            uncertain_resp_index = [int(feed_dict["responses"][i]) for i in uncertain_index]
            self.available_cand_index = list(set(self.available_cand_index) | set(uncertain_resp_index))
            self.available_cand_index.sort()
            current_candidates_rep = candidates_rep[self.available_cand_index]

            # step2: Get the uncertain cond & resp embedding
            uncertain_cond_embedding = cond_embedding[uncertain_index]
            uncertain_resp_embedding = candidates_rep[uncertain_resp_index]

            # step3: fuse cond and resp
            recog_input = torch.cat([uncertain_cond_embedding, uncertain_resp_embedding], 1)

            # step4: Get z posterior
            posterior_mulogvar = self.recogNet_mulogvar(recog_input)
            posterior_mu, posterior_logvar = torch.chunk(posterior_mulogvar, 2, 1)
            # todo: sample more posterior may increase data efficiency
            latent_posterior = sample_gaussian(posterior_mu, posterior_logvar, 1).squeeze(1)

            # step5: Get loss
            cond_z_embed_posterior = self.fused_cond_z(torch.cat([uncertain_cond_embedding, latent_posterior], 1))
            uncertain_logits = torch.matmul(cond_z_embed_posterior, current_candidates_rep.t())
            target = list(map(lambda resp_index: self.available_cand_index.index(resp_index), uncertain_resp_index))
            target = torch.Tensor(target).to(uncertain_logits.device, dtype=torch.long)
            # todo: maybe other loss form for data recover such as: max margin
            avg_rc_loss = F.cross_entropy(uncertain_logits, target)
            kld = gaussian_kld(posterior_mu, posterior_logvar,
                               prior_mu[uncertain_index], prior_logvar[uncertain_index])
            avg_kld = torch.mean(kld)
            # todo: KL weight
            kl_weights = 1.0
            elbo = avg_rc_loss + kl_weights * avg_kld
            # todo: more loss. THIS IS VERY IMPORTANT
            # todo: Such as mutual information to stable z, weight lock for continuous learning, normalisation term
        else:
            elbo = None
        return elbo, \
               uncertain_index, \
               certain_index, \
               certain_response, \
               elbo.item() if elbo is not None else 0, \
               avg_rc_loss.item() if elbo is not None else 0, \
               avg_kld.item() if elbo is not None else 0, \
               acc


class ContinuousAgent(object):
    def __init__(self, config, model, api):
        np.random.seed(config["random_seed"] + 1)
        self.config = config
        self.model = model.to(config["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        self.api = api
        self.coming_data = list()
        self.test_data = dict()
        for task in TASKS.keys():
            for data_set in ["train", "dev"]:
                self.coming_data.extend(self.api.all_data[task][data_set])
        self.comingS, self.comingQ, self.comingA = self.api.vectorize_data(self.coming_data, self.config["batch_size"])
        for task in TASKS.keys():
            self.test_data[task] = dict()
            self.test_data[task]["S"], self.test_data[task]["Q"], self.test_data[task]["A"] = self.api.vectorize_data(
                self.api.all_data[task]["test"], self.config["batch_size"])

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

    def simulate_run(self):
        loss_log = defaultdict(list)

        self.logger.info("Run main with config {}".format(self.config))

        for s, q, a in batch_iter(self.comingS, self.comingQ, self.comingA, self.config["batch_size"], shuffle=True):
            self.optimizer.zero_grad()
            feed_dict = {"contexts": (self.tensor_wrapper(s), self.tensor_wrapper(q)),
                         "responses": a}
            elbo, uncertain_index, certain_index, certain_response, elbo_item, avg_rc_loss_item, avg_kld_item, acc = self.model.forward(
                feed_dict)
            if elbo is not None:
                elbo.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_clip"])
                self.optimizer.step()
            print(len(certain_index), elbo_item, avg_rc_loss_item, avg_kld_item, acc)
            loss_log["certain"].append(len(certain_index))
            loss_log["elbo"].append(elbo_item)
            loss_log["avg_rc_loss"].append(avg_rc_loss_item)
            loss_log["avg_kld_loss"].append(avg_kld_item)
            loss_log["acc"].append(acc)

        torch.save(self.model.state_dict(), os.path.join(self.config["save_dir"], "model.pkl"))
        pickle.dump(loss_log, open(os.path.join("debug", "loss.log"), "wb"))