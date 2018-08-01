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
from Continuous_VAE.model.utils import sample_gaussian, gaussian_kld
from Continuous_VAE.data_apis.data_utils import batch_iter, TASKS, DATA_ROOT
from nn_utils.nn_utils import Attn, bow_sentence, bow_sentence_self_attn, rnn_seq, rnn_seq_self_attn, RnnV, \
    SelfAttn


class ContinuousVAE(nn.Module):
    def __init__(self, config, api):
        super(ContinuousVAE, self).__init__()
        torch.manual_seed(config.random_seed)

        self.api = api
        self.config = config

        self.embedding = nn.Embedding(self.api.vocab_size, self.config.word_emb_size, padding_idx=0)

        if self.config.sent_encode_method == "rnn":
            self.sent_rnn = RnnV(self.config.word_emb_size, self.config.sent_rnn_hidden_size,
                                 self.config.sent_rnn_type, self.config.sent_rnn_layers,
                                 dropout=self.config.sent_rnn_dropout,
                                 bidirectional=self.config.sent_rnn_bidirectional)
        if self.config.sent_self_attn is True:
            self.sent_self_attn_layer = SelfAttn(self.config.sent_emb_size,
                                                 self.config.sent_self_attn_hidden,
                                                 self.config.sent_self_attn_head)

        if self.config.ctx_encode_method == "MemoryNetwork":
            self.attn_layer = Attn(self.config.attn_method, self.config.sent_emb_size, self.config.sent_emb_size)
            self.hops_map = nn.Linear(self.config.sent_emb_size, self.config.sent_emb_size)
            if self.config.memory_nonlinear.lower() == "tanh":
                self.memory_nonlinear = nn.Tanh()
            elif self.config.memory_nonlinear.lower() == "relu":
                self.memory_nonlinear = nn.ReLU()
            elif self.config.memory_nonlinear.lower() == "iden":
                self.memory_nonlinear = None
        elif self.config.ctx_encode_method == "HierarchalRNN":
            self.ctx_rnn = RnnV(self.config.sent_emb_size, self.config.ctx_rnn_hidden_size,
                                self.config.ctx_rnn_type, self.config.ctx_rnn_layers,
                                dropout=self.config.ctx_rnn_dropout,
                                bidirectional=self.config.ctx_rnn_bidirectional)
        elif self.config.ctx_self_attn is True or self.config.ctx_encode_method == "HierarchalSelfAttn":
            self.ctx_self_attn_layer = SelfAttn(self.config.ctx_emb_size,
                                                self.config.ctx_self_attn_hidden,
                                                self.config.ctx_self_attn_head)

        # todo: add more in cond
        cond_emb_size = self.config.ctx_emb_size
        response_emb_size = self.config.sent_emb_size
        recog_input_size = cond_emb_size + response_emb_size

        # todo: make prior and posterior more expressive (IAF)
        # recognitionNetwork: A MLP
        self.recogNet_mulogvar = nn.Sequential(
            nn.Linear(recog_input_size, max(50, self.config.latent_size * 2)),
            nn.Tanh(),
            nn.Linear(max(50, self.config.latent_size * 2), self.config.latent_size * 2)
        )

        # priorNetwork: A MLP
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_emb_size, max(50, self.config.latent_size * 2)),
            nn.Tanh(),
            nn.Linear(max(50, self.config.latent_size * 2), self.config.latent_size * 2)
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
        # todo: only for test!!!
        # for task in ["task_1", "task_2", "task_3", "task_4", "task_5"]:
        #     with open(os.path.join(DATA_ROOT, "candidate", task + ".txt")) as f:
        #         for line in f:
        #             line = line.strip()
        #             if api.candid2index[line] not in self.available_cand_index:
        #                 self.available_cand_index.append(api.candid2index[line])
        self.available_cand_index.sort()
        self.register_buffer("candidates", torch.from_numpy(api.vectorize_candidates()))

        # fuse cond_embed and z
        # todo: there may be other method
        # self.fused_cond_z = nn.Sequential(
        #     nn.Linear(cond_emb_size + self.config.latent_size, self.config.sent_emb_size),
        #     nn.Tanh(),
        #     nn.Linear(self.config.sent_emb_size, self.config.sent_emb_size),
        # )
        self.fused_cond_z = nn.Linear(cond_emb_size + self.config.latent_size, self.config.sent_emb_size)

        # todo: pretend z vanish
        self.drop = nn.Dropout(p=0.5)

        # todo: there may be other method for scoring function
        # self.score = nn.Sequential(
        #     nn.Linear(self.config.sent_emb_size * 2, self.config.sent_emb_size),
        #     nn.Tanh(),
        #     nn.Linear(self.config.sent_emb_size, 1)
        # )

        # for debug
        self.error = defaultdict(list)

    def sent_encode(self, s):
        if self.config.sent_encode_method == "bow":
            if self.config.sent_self_attn is False:
                s_encode = bow_sentence(self.embedding(s), self.config.emb_sum)
            else:
                s_encode = bow_sentence_self_attn(self.embedding(s), self.sent_self_attn_layer)
        elif self.config.sent_encode_method == "rnn":
            if self.config.sent_self_attn is False:
                s_encode = rnn_seq(self.embedding(s), self.sent_rnn, self.config.sent_emb_size)
            else:
                s_encode = rnn_seq_self_attn(self.embedding(s), self.sent_rnn,
                                             self.sent_self_attn_layer, self.config.sent_emb_size)
        return s_encode

    def ctx_encode_m2n(self, contexts):
        stories, queries = contexts
        m = self.sent_encode(stories)
        q = self.sent_encode(queries)

        u = [q]

        for _ in range(self.config.max_hops):
            # attention over memory and read memory
            _, o_k = self.attn_layer(m, u[-1])

            # fuse read memory and previous hops
            u_k = self.hops_map(u[-1]) + o_k
            if self.memory_nonlinear is not None:
                u_k = self.memory_nonlinear(u_k)

            u.append(u_k)

        return u[-1]

    def ctx_encode_h_self_attn(self, contexts):
        stories, _ = contexts
        m = self.sent_encode(stories)
        return self.ctx_self_attn_layer(m)

    def ctx_encode_h_rnn(self, contexts):
        stories, _ = contexts
        m = self.sent_encode(stories)

        if self.config.ctx_self_attn is True:
            return rnn_seq_self_attn(m, self.ctx_rnn, self.ctx_self_attn_layer, self.config.ctx_emb_size)
        else:
            return rnn_seq(m, self.ctx_rnn, self.config.ctx_emb_size)

    def threshold_method(self, sampled_response):
        uncertain_index = list()
        certain_index = list()
        certain_response = list()

        # todo: can we accelerate this part in GPU?
        for i, response_dist in enumerate(sampled_response):
            vot_result, vot_num = Counter(response_dist).most_common(1)[0]
            if vot_num < self.config.threshold * self.config.prior_sample:
                uncertain_index.append(i)
            else:
                certain_index.append(i)
                certain_response.append(self.available_cand_index[vot_result])

        return uncertain_index, certain_index, certain_response

    def select_uncertain_points(self, logits):
        probs = F.softmax(logits, 2)
        probs = probs.contiguous().view(-1, len(self.available_cand_index))
        sampled_response = torch.multinomial(probs, 1)
        sampled_response = sampled_response.view(-1, self.config.prior_sample)
        sampled_response = sampled_response.detach().data.cpu().numpy()

        # todo: more heuristic method for uncertain points
        uncertain_index, certain_index, certain_response = self.threshold_method(sampled_response)

        return uncertain_index, certain_index, certain_response

    def evaluate(self, certain_index, certain_responses, feed_dict):
        feed_responses = np.array([feed_dict["responses"][i] for i in certain_index])
        certain_responses = np.array(certain_responses)
        certain_index = np.array(certain_index)
        acc = metrics.accuracy_score(feed_responses, certain_responses)

        if acc < 1:
            for certain_idx, certain_response, feed_response in zip(
                    certain_index[np.not_equal(feed_responses, certain_responses)],
                    certain_responses[np.not_equal(feed_responses, certain_responses)],
                    feed_responses[np.not_equal(feed_responses, certain_responses)]):
                self.error[" ".join(self.api.candidates[feed_response])].append(
                    (self.helper(feed_dict["contexts"][0][certain_idx]),
                     self.helper(feed_dict["contexts"][1][certain_idx]),
                     " ".join(self.api.candidates[certain_response]),
                     feed_dict["step"],
                     certain_idx))

        return acc

    def helper(self, s):
        s = s.data.cpu().numpy()
        string = ""
        if s.ndim == 2:
            for l in s:
                l = list(filter(lambda x: x != 0, l))
                l = " ".join([self.api.index2word[x] for x in l])
                string += l
                string += "\n"
        else:
            l = list(filter(lambda x: x != 0, s))
            l = " ".join([self.api.index2word[x] for x in l])
            string += l
            string += "\n"
        return string

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config.device)

    def forward(self, feed_dict):
        """
        Step1: Get the context representation
        Step2: Sample z from prior
        Step3: Get the slice of uncertain points and certain points
        Step4: Calculate the IR Evaluation on the certain points and uncertain points
        Step5: Get the loss
        """
        # Step1: Get the context representation
        # todo: more complex method for sentence embed
        if self.config.ctx_encode_method == "MemoryNetwork":
            context_rep = self.ctx_encode_m2n(feed_dict["contexts"])
        elif self.config.ctx_encode_method == "HierarchalSelfAttn":
            context_rep = self.ctx_encode_h_self_attn(feed_dict["contexts"])
        elif self.config.ctx_encode_method == "HierarchalRNN":
            context_rep = self.ctx_encode_h_rnn(feed_dict["contexts"])

        # todo: we may add more to cond_embed
        cond_emb = context_rep

        # Step2: Sample z from prior
        prior_mulogvar = self.priorNet_mulogvar(cond_emb)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
        latent_prior = sample_gaussian(prior_mu, prior_logvar, self.config.prior_sample)

        # Step3: Get the slice of uncertain points and certain points
        #    step1: fusing z and cond_embed
        #    step2: Get the embed of current candidates
        #    step3: Get the logits of each candidate
        #    step4: A method to judge which context is uncertain

        # step1: fusing z and cond_embed
        # todo: The z can be seen as a gate or other complex method
        # todo: Now we do it very naive
        cond_emb_temp = cond_emb.unsqueeze(1).expand(-1, self.config.prior_sample, -1)
        cond_z_embed_prior = self.fused_cond_z(torch.cat([cond_emb_temp, latent_prior], 2))
        # step2: Get the embed of current candidates
        candidates_rep = self.sent_encode(self.candidates)
        current_candidates_rep = candidates_rep[self.available_cand_index]
        # step3: Get the logits of each candidate
        # todo: more complicated methods for candidates scoring
        logits = torch.matmul(cond_z_embed_prior, current_candidates_rep.t())
        # logits = self.score(
        #     torch.cat([cond_z_embed_prior.unsqueeze(2).expand(-1, -1, current_candidates_rep.size(0), -1),
        #                current_candidates_rep.expand(cond_z_embed_prior.size(0), cond_z_embed_prior.size(1), -1, -1)],
        #               3)).squeeze(3)
        # step4: A method to judge which context is uncertain
        uncertain_index, certain_index, certain_response = self.select_uncertain_points(logits)

        # Step4: Calculate the IR Evaluation on the certain points and uncertain points
        # todo: we left this part aside
        if len(certain_index) > 0:
            acc = self.evaluate(certain_index, certain_response, feed_dict)
        else:
            acc = 0

        # Step5: Get the loss
        #    step1: Simulate human in the loop and update the available response set
        #    step2: Get the uncertain cond & resp embed
        #    step3: fuse cond & resp
        #    step4: Get z posterior
        #    step5: fuse z and cond

        if len(uncertain_index) > 0:
            # step1: Simulate human in the loop and update the available response set
            uncertain_resp_index = [int(feed_dict["responses"][i]) for i in uncertain_index]
            self.available_cand_index = list(set(self.available_cand_index) | set(uncertain_resp_index))
            self.available_cand_index.sort()
            current_candidates_rep = candidates_rep[self.available_cand_index]

            # # add all tagged data
            # for position, resp in zip(uncertain_index, uncertain_resp_index):
            #     self.api.tagged[resp].append(feed_dict["start"] + position)

            # step2: Get the uncertain cond & resp embed
            uncertain_cond_emb = cond_emb[uncertain_index]
            uncertain_resp_emb = candidates_rep[uncertain_resp_index]

            # step3: fuse cond and resp
            recog_input = torch.cat([uncertain_cond_emb, uncertain_resp_emb], 1)

            # step4: Get z posterior
            posterior_mulogvar = self.recogNet_mulogvar(recog_input)
            posterior_mu, posterior_logvar = torch.chunk(posterior_mulogvar, 2, 1)
            # todo: sample more posterior may increase data efficiency
            latent_posterior = sample_gaussian(posterior_mu, posterior_logvar, self.config.posterior_sample)

            # step5: Get loss
            uncertain_cond_emb_temp = uncertain_cond_emb.unsqueeze(1).expand(-1, self.config.posterior_sample, -1)
            cond_z_embed_posterior = self.fused_cond_z(
                torch.cat([self.drop(uncertain_cond_emb_temp), latent_posterior], 2))
            uncertain_logits = torch.matmul(cond_z_embed_posterior, current_candidates_rep.t()).contiguous()
            uncertain_logits = uncertain_logits.view(-1, uncertain_logits.size(2))
            # uncertain_logits = self.score(
            #     torch.cat([cond_z_embed_posterior.unsqueeze(1).expand(-1, current_candidates_rep.size(0), -1),
            #                current_candidates_rep.expand(cond_z_embed_posterior.size(0), -1, -1)], 2)).squeeze(2)

            target = list(map(lambda resp_index: self.available_cand_index.index(resp_index), uncertain_resp_index))
            target = torch.Tensor(target).to(uncertain_logits.device, dtype=torch.long)
            target = target.unsqueeze(1).expand(-1, self.config.posterior_sample).contiguous().view(-1)

            # todo: maybe other loss form for data recover such as: max margin
            avg_rc_loss = F.cross_entropy(uncertain_logits, target)

            kld = gaussian_kld(posterior_mu, posterior_logvar,
                               prior_mu[uncertain_index], prior_logvar[uncertain_index])
            avg_kld = torch.mean(kld)
            # # todo: KL weight
            kl_weights = feed_dict["step"] / self.config.full_kl_step

            # # todo: add cluster loss
            # # We add reg term in posterior!!!!!
            # # same response should have same posterior
            # posterior_mu_temp = posterior_mu.unsqueeze(1).expand(-1, self.config.expect_sample, -1).contiguous().view(-1, posterior_mu.size(-1))
            # posterior_logvar_temp = posterior_logvar.unsqueeze(1).expand(-1, self.config.expect_sample, -1).contiguous().view(-1, posterior_logvar.size(-1))
            #
            # positive_samples = list()
            # for resp_c in uncertain_resp_index:
            #     positive_samples.extend(self.api.sample_true(resp_c, self.config.expect_sample))
            #
            # positive_samples_ctx = (self.tensor_wrapper([self.api.comingS[i] for i in positive_samples]),
            #                         self.tensor_wrapper([self.api.comingQ[i] for i in positive_samples]))
            # positive_samples_ctx_encode = self.ctx_encode_m2n(positive_samples_ctx)
            # positive_samples_resp_encode = uncertain_resp_emb.unsqueeze(1).expand(-1, self.config.expect_sample, -1).contiguous().view(-1, uncertain_resp_emb.size(-1))
            # positive_samples_recog_input = torch.cat([positive_samples_ctx_encode, positive_samples_resp_encode], 1)
            # positive_samples_posterior_mulogvar = self.recogNet_mulogvar(positive_samples_recog_input)
            # positive_samples_posterior_mu, positive_samples_posterior_logvar = torch.chunk(positive_samples_posterior_mulogvar, 2, 1)
            # kld_same = gaussian_kld(posterior_mu_temp, posterior_logvar_temp,
            #                         positive_samples_posterior_mu, positive_samples_posterior_logvar).mean()
            # kld_same += gaussian_kld(positive_samples_posterior_mu, positive_samples_posterior_logvar,
            #                          posterior_mu_temp, posterior_logvar_temp).mean()
            # kld_same *= 0.5
            #
            # # different response should have different posterior
            # negative_samples = list()
            # for resp_c in uncertain_resp_index:
            #     negative_samples.extend(self.api.sample_negative(resp_c, self.config.expect_sample))
            #
            # negative_samples_ctx = (self.tensor_wrapper([self.api.comingS[i] for i in negative_samples]),
            #                         self.tensor_wrapper([self.api.comingQ[i] for i in negative_samples]))
            # negative_samples_ctx_encode = self.ctx_encode_m2n(negative_samples_ctx)
            # negative_samples_resp_encode = candidates_rep[[int(self.api.comingA[i]) for i in negative_samples]]
            # negative_samples_recog_input = torch.cat([negative_samples_ctx_encode, negative_samples_resp_encode], 1)
            # negative_samples_posterior_mulogvar = self.recogNet_mulogvar(negative_samples_recog_input)
            # negative_samples_posterior_mu, negative_samples_posterior_logvar = torch.chunk(negative_samples_posterior_mulogvar, 2, 1)
            # kld_diff = gaussian_kld(posterior_mu_temp, posterior_logvar_temp, negative_samples_posterior_mu, negative_samples_posterior_logvar).mean()
            # kld_diff += gaussian_kld(negative_samples_posterior_mu, negative_samples_posterior_logvar, posterior_mu_temp, posterior_logvar_temp).mean()
            # kld_diff *= 0.5

            # # todo: add cluster loss
            # We add reg term in prior!!!!!
            # same response should have same posterior
            # prior_mu_uncertain_temp = prior_mu[uncertain_index].unsqueeze(1).expand(-1, self.config.expect_sample, -1).contiguous().view(-1, prior_mu.size(-1))
            # prior_logvar_uncertain_temp = prior_logvar[uncertain_index].unsqueeze(1).expand(-1, self.config.expect_sample,-1).contiguous().view(-1, prior_logvar.size(-1))
            #
            # positive_samples = list()
            # for resp_c in uncertain_resp_index:
            #     positive_samples.extend(self.api.sample_true(resp_c, self.config.expect_sample))
            #
            # positive_samples_ctx = (self.tensor_wrapper([self.api.comingS[i] for i in positive_samples]),
            #                         self.tensor_wrapper([self.api.comingQ[i] for i in positive_samples]))
            # positive_samples_ctx_encode = self.ctx_encode_m2n(positive_samples_ctx)
            # positive_samples_resp_encode = uncertain_resp_emb.unsqueeze(1).expand(-1, self.config.expect_sample, -1).contiguous().view(-1, uncertain_resp_emb.size(-1))
            # positive_samples_recog_input = torch.cat([positive_samples_ctx_encode, positive_samples_resp_encode], 1)
            # positive_samples_posterior_mulogvar = self.recogNet_mulogvar(positive_samples_recog_input)
            # positive_samples_posterior_mu, positive_samples_posterior_logvar = torch.chunk(positive_samples_posterior_mulogvar, 2, 1)
            # kld_same = gaussian_kld(positive_samples_posterior_mu, positive_samples_posterior_logvar, prior_mu_uncertain_temp, prior_logvar_uncertain_temp).mean()

            # # different response should have different posterior
            # negative_samples = list()
            # for resp_c in uncertain_resp_index:
            #     negative_samples.extend(self.api.sample_negative(resp_c, self.config.expect_sample))
            #
            # negative_samples_ctx = (self.tensor_wrapper([self.api.comingS[i] for i in negative_samples]),
            #                         self.tensor_wrapper([self.api.comingQ[i] for i in negative_samples]))
            # negative_samples_ctx_encode = self.ctx_encode_m2n(negative_samples_ctx)
            # negative_samples_resp_encode = candidates_rep[[int(self.api.comingA[i]) for i in negative_samples]]
            # negative_samples_recog_input = torch.cat([negative_samples_ctx_encode, negative_samples_resp_encode], 1)
            # negative_samples_posterior_mulogvar = self.recogNet_mulogvar(negative_samples_recog_input)
            # negative_samples_posterior_mu, negative_samples_posterior_logvar = torch.chunk(
            #     negative_samples_posterior_mulogvar, 2, 1)
            # kld_diff = gaussian_kld(prior_mu_uncertain_temp, prior_logvar_uncertain_temp,
            #                         negative_samples_posterior_mu,negative_samples_posterior_logvar).mean()
            # kld_diff += gaussian_kld(negative_samples_posterior_mu, negative_samples_posterior_logvar,
            #                          prior_mu_uncertain_temp, prior_logvar_uncertain_temp).mean()
            # kld_diff *= 0.5

            elbo = avg_rc_loss + avg_kld * kl_weights
            # todo: add MI loss
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
        np.random.seed(config.random_seed + 1)
        self.config = config
        self.model = model.to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.api = api
        self.coming_data = list()
        self.test_data = dict()
        for task in TASKS.keys():
            for data_set in ["train", "dev"]:
                self.coming_data.extend(self.api.all_data[task][data_set])
        self.comingS, self.comingQ, self.comingA = self.api.vectorize_data(self.coming_data, self.config.batch_size)
        # self.api.comingS = self.comingS
        # self.api.comingQ = self.comingQ
        # self.api.comingA = self.comingA
        for task in TASKS.keys():
            self.test_data[task] = dict()
            self.test_data[task]["S"], self.test_data[task]["Q"], self.test_data[task]["A"] = self.api.vectorize_data(
                self.api.all_data[task]["test"], self.config.batch_size)

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config.device)

    def simulate_run(self):
        loss_log = defaultdict(list)

        for step, (s, q, a, start) in enumerate(
                batch_iter(self.comingS, self.comingQ, self.comingA, self.config.batch_size, shuffle=True)):
            # print("step:", step)
            self.optimizer.zero_grad()
            feed_dict = {"contexts": (self.tensor_wrapper(s), self.tensor_wrapper(q)),
                         "responses": a, "step": step}
            elbo, uncertain_index, certain_index, certain_response, elbo_item, avg_rc_loss_item, avg_kld_item, acc = \
                self.model(feed_dict)
            if elbo is not None:
                elbo.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_clip)
                self.optimizer.step()
            print(len(certain_index), elbo_item, avg_rc_loss_item, avg_kld_item, acc)
            # print(len(certain_index), acc)
            loss_log["certain"].append(len(certain_index))
            loss_log["elbo"].append(elbo_item)
            loss_log["avg_rc_loss"].append(avg_rc_loss_item)
            loss_log["avg_kld_loss"].append(avg_kld_item)
            loss_log["acc"].append(acc)

        torch.save(self.model.state_dict(), self.config.model_save_path)
        pickle.dump(loss_log, open(self.config.debug_path, "wb"))

        # for debug
        print("debug details")
        for feed_response, contents in self.model.error.items():
            print("**************************************************************************************")
            print(">>>>", feed_response)
            for content in contents:
                for c in content:
                    print(c)
                print("\n")
            print("**************************************************************************************")
