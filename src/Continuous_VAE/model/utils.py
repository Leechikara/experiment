# coding=utf-8
import numpy as np
import torch


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = - 0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                            - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                            - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5 * torch.sum(logvar + np.log(2 * np.pi) + torch.div(torch.pow((x - mu), 2), torch.exp(logvar)), 1)


def sample_gaussian(mu, logvar, n):
    # return (batch, n, z_dim)
    eps_shape = list(mu.shape)
    eps_shape.insert(1, n)
    mu_temp = torch.unsqueeze(mu, 1)
    logvar_temp = torch.unsqueeze(logvar, 1)
    epsilon = torch.randn(eps_shape, device=mu.device)
    std = torch.exp(0.5 * logvar_temp)
    z = mu_temp + std * epsilon
    return z
