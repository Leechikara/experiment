# coding = utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# sentence embedding method
def get_bow(embedding, avg=True):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length.
    The rank must be not less than 3.
    """
    embedding_size = embedding.size(-1)
    if avg:
        return embedding.mean(-2), embedding_size
    else:
        return embedding.sum(-2), embedding_size


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        assert method in ["dotted", "general", "concat"]
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, encode_outputs, decode_state):
        # todo: add mask
        """
        :param encode_outputs: (batch, output_length, hidden_size)
        :param decode_state: (batch, hidden_size)
        :return: (batch, output_length)
        """
        output_length = encode_outputs.size(1)
        if self.method == "concat":
            decode_state_temp = decode_state.unsqueeze(1)
            decode_state_temp = decode_state_temp.expand(-1, output_length, -1)
            cat_encode_decode = torch.cat([encode_outputs, decode_state_temp], 2)
            energy = self.attn(cat_encode_decode).squeeze(-1)
        elif self.method == "general":
            decode_state_temp = decode_state.unsqueeze(1)
            mapped_encode_outputs = self.attn(encode_outputs)
            energy = torch.sum(decode_state_temp * mapped_encode_outputs, 2)
        else:
            decode_state_temp = decode_state.unsqueeze(1)
            energy = torch.sum(decode_state_temp * encode_outputs, 2)
        probs = F.softmax(energy, dim=1)
        return probs
