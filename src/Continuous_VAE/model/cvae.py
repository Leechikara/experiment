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

sys.path.append("/home/wkwang/workstation/experiment/src")
from Continuous_VAE.model.utils import sample_gaussian, norm_log_liklihood, gaussian_kld
from Continuous_VAE.data_apis.data_utils import batch_iter, DataUtils, TASKS


class ContinuousVAE(nn.Module):
    def __init__(self, config, api):
        super(ContinuousVAE, self).__init__()
        torch.manual_seed(config["random_seed"])

        self.config = config
        self.vocab_size = api.vocab_size
        self.embed_dim = config["embed_dim"]

        # Embedding for context and response
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # todo: Encoding method for context, Now we only implement the basic method
        # todo: Encoding method for response, Now we only implement the basic method
        if config["context_encoding_method"] == "MemoryNetwork":
            self.hops_map = nn.Linear(self.embed_dim, self.embed_dim)
            assert config["memory_nonlinear"].lower() in ["tanh", "iden", "relu"]
            if config["memory_nonlinear"].lower() == "tanh":
                self.memory_nonlinear = nn.Tanh()
            elif config["memory_nonlinear"].lower() == "relu":
                self.memory_nonlinear == nn.ReLU()
            else:
                self.memory_nonlinear = None

        cond_embedding_size = self.embed_dim
        response_embedding_size = self.embed_dim
        recog_input_size = cond_embedding_size + response_embedding_size

        # recognitionNetwork: A MLP
        self.recogNet_mulogvar = nn.Sequential(
            nn.Linear(recog_input_size, np.maximum(50, config["latent_size"] * 2)),
            nn.Tanh(),
            nn.Linear(np.maximum(50, config["latent_size"] * 2), config["latent_size"] * 2)
        )

        # priorNetwork: A MLP
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_embedding_size, np.maximum(50, config["latent_size"] * 2)),
            nn.Tanh(),
            nn.Linear(np.maximum(50, config["latent_size"] * 2), config["latent_size"] * 2)
        )

        # record all candidates in advance and current available index
        # If new candidates are known, we add its index into  available_cand_index
        # I think it's a more efficient strategy to simulate continuous leaning
        self.available_cand_index = list()
        self.register_buffer("candidates", torch.from_numpy(api.candidates))

        # fuse cond_embedding and z
        # todo: there may be other method
        self.fused_cond_z = nn.Linear(self.embed_dim + config["latent_size"], self.embed_dim)

    def context_encoding_m2n(self, contexts):
        # encoding the contexts in a batch by memory networks
        stories, queries = contexts
        m = self.embedding(stories).sum(2)
        q = self.embedding(queries).sum(1)
        u = [q]

        for _ in range(self.config["max_hops"]):
            # attention over memory
            u_temp = torch.unsqueeze(u[-1], 1)
            dotted = torch.sum(m * u_temp, 2)
            probs = F.softmax(dotted, dim=1)

            # read memory
            probs_temp = torch.unsqueeze(probs, 1)
            c_temp = torch.transpose(m, 1, 2)
            o_k = torch.sum(c_temp * probs_temp, 2)

            # fuse read memory and previous hops
            u_k = self.hops_map(u[-1]) + o_k
            if self.config["memory_nonlinear"] is not None:
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
            if vot_num < self.config["threshold"] * len(self.available_cand_index):
                uncertain_index.append(i)
            else:
                certain_index.append(i)
                certain_response.append(self.available_cand_index[vot_result])

        return uncertain_index, certain_index, certain_response

    def select_uncertain_points(self, logits):
        probs = F.softmax(logits, 2)
        probs = probs.contiguous().view(-1, len(self.available_cand_index))
        sampled_response = torch.multinomial(probs, 1)
        sampled_response = sampled_response.view(-1, self.config["n_forward"])
        sampled_response = sampled_response.detach().data.cpu().numpy()

        # todo: more heuristic method for uncertain points
        uncertain_index, certain_index, certain_response = self.threshold_method(sampled_response)

        return uncertain_index, certain_index, certain_response

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

        # todo: we may add something to cond_embedding
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
        candidates_rep = self.embedding(self.candidates).sum(1)
        current_candidates_rep = candidates_rep[self.available_cand_index]
        # step3: Get the logits of each candidate
        # todo: more complicated methods for candidates scoring
        logits = torch.matmul(cond_z_embed_prior, current_candidates_rep.t())
        # step4: A method to judge which context is uncertain
        uncertain_index, certain_index, certain_response = self.select_uncertain_points(logits)

        # Step4: Calculate the IR Evaluation on the certain points and uncertain points
        # todo: we left this part aside

        # Step5: Get the loss
        #    step1: Simulate human in the loop and update the available response set
        #    step2: Get the uncertain cond & resp embedding
        #    step3: fuse cond & resp
        #    step4: Get z posterior
        #    step5: fuse z and cond

        if len(uncertain_index) > 0:
            # step1: Simulate human in the loop and update the available response set
            uncertain_resp_index = feed_dict["responses"][uncertain_index]
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
            latent_posterior = sample_gaussian(posterior_mu, posterior_logvar, 1).squeeze()

            # step5: Get loss
            cond_z_embed_posterior = self.fused_cond_z(torch.cat([uncertain_cond_embedding, latent_posterior], 1))
            uncertain_logits = torch.matmul(cond_z_embed_posterior, current_candidates_rep.t())
            target = list(map(lambda resp_index: self.available_cand_index.index(resp_index), uncertain_resp_index))
            target = torch.LongTensor(target, device=uncertain_logits.device)
            # todo: maybe other loss form for data recover such as: max margin
            avg_rc_loss = F.cross_entropy(uncertain_logits, target)
            kld = gaussian_kld(posterior_mu, posterior_mulogvar,
                               prior_mu[uncertain_index], prior_logvar[uncertain_index])
            avg_kld = torch.mean(kld)
            # todo: KL weight
            kl_weights = 1.0
            elbo = avg_rc_loss + kl_weights * avg_kld
            # todo: more loss. THIS IS VERY IMPORTANT
            # todo: Such as mutual information to stable z, weight lock for continuous learning, normalisation term
        else:
            elbo = None
        return elbo, uncertain_index, certain_index, certain_response


class ContinuousAgent(object):
    def __init__(self, config, model, api):
        np.random.seed(config["random_seed"] + 1)
        self.config = config
        self.model = model.to(config["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))
        self.api = api
        coming_data = list()
        self.test_data = dict()
        for task in TASKS.keys():
            for data_set in ["train", "dev"]:
                coming_data.extend(self.api.all_data[task][data_set])
        self.comingS, self.comingQ, self.comingA = self.api.vectorize_data(coming_data, self.config["batch_size"])
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
        self.logger.info("Run main with config {}".format(self.config))

        for s, q, a in batch_iter(self.comingS, self.comingQ, self.comingA, self.config["batch_size"], shuffle=True):
            self.optimizer.zero_grad()
            feed_dict = {"contexts": (self.tensor_wrapper(s), self.tensor_wrapper(q)),
                         "responses": self.tensor_wrapper(a)}
            elbo, uncertain_index, certain_index, certain_response = self.model.forward(feed_dict)
            if elbo is not None:
                elbo.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_clip"])
                self.optimizer.step()
