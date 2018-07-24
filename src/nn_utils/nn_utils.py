# coding = utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# sentence embedding method
def bow_sentence(input, emb_sum=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length.
    The rank must be not less than 3.(batch, *, seq_len, feature)
    """
    embedding_size = input.size(-1)
    if emb_sum:
        return input.sum(-2), embedding_size
    else:
        return input.mean(-2), embedding_size


def bow_sentence_self_att(input, self_att_model):
    """
    :param input: (batch, seq_len, feature)
    :param self_att_model: A object of SelfAttn
    :return: (batch, feature)
    """
    return self_att_model(input)


def rnn_sentence(input, sent_len, rnn_model):
    """
    :param input: (batch, seq_len, feature)
    :param sent_len: (batch,)
    :param rnn_model: A object of RnnV
    :return: The last hidden state of RNN model (batch, feature)
    """
    h_n = rnn_model(input, sent_len)
    h_n = h_n.transpose(0, 1)
    h_n = h_n.contiguous().view(h_n.size(0), -1)
    return h_n


def rnn_sentence_self_att(input, sent_len, rnn_model, self_att_model):
    """
    :param input: (batch, seq_len, feature)
    :param sent_len: (batch,)
    :param rnn_model: A object of RnnV
    :param self_att_model: A object of SelfAttn
    :return: Apply rnn model on input and use self attention
    """
    (out, _), _ = rnn_model(input, sent_len, False)
    return self_att_model(out)


class SelfAttn(nn.Module):
    def __init__(self, input_dim, hidden, head):
        super(SelfAttn, self).__init__()

        self.linear_first = nn.Linear(input_dim, hidden)
        self.linear_second = nn.Linear(hidden, head)
        self.linear_first.bias.data.fill_(0)
        self.linear_second.bias.data.fill_(0)
        self.head = head

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, 1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, rnn_output):
        """
        Do self attention over the output of RNN
        :param rnn_output: (batch, seq_len, feature)
        :return: (batch, feature)
        """
        x = F.tanh(self.linear_first(rnn_output))
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = torch.matmul(attention, rnn_output)
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.head
        return avg_sentence_embeddings


class RnnV(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type="gru",
                 num_layers=1,
                 bias=True,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False):
        """
        RNN which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param rnn_type: The RNN type, Default: gru
        :param num_layers: Number of recurrent layers
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True (default True), the input and output tensors are provided as (batch, seq, feature)
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(RnnV, self).__init__()

        try:
            assert rnn_type.lower() in ['gru', 'lstm']
        except TypeError:
            print("Expect rnn type is GRU or LSTM, but accept " + rnn_type)
            exit()

        self.rnn_type = rnn_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.directions = 2 if bidirectional else 1

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional)

    def forward(self, x, x_len, only_use_last_hidden_state=True):
        """
        sequence -> sort -> pack -> process using RNN -> unpack -> unsort

        :param x: If batch_first is True(default), then Variable are provided as (batch, seq_len, input_size)
        :param x_len: valid seq_len numpy list in a batch, provided as (batch, )
        :param only_use_last_hidden_state: return h_n if true
        :return:
        """
        # obtain sort index and unsort index
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = np.argsort(x_sort_idx)

        # sort x and x_len in batch dimension
        x = x.select(0 if self.batch_first else 1, x_sort_idx)
        x_len = x_len[x_sort_idx]

        # pack x
        x_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using RNN
        h_0 = torch.zeros(self.num_layers * self.directions,
                          x.size(0) if self.batch_first else x.size(1),
                          self.hidden_size).to(x.device)
        if self.rnn_type == "lstm":
            c_0 = torch.zeros(self.num_layers * self.directions,
                              x.size(0) if self.batch_first else x.size(1),
                              self.hidden_size).to(x.device)
            out_pack, (h_n, c_n) = self.rnn(x_p, (h_0, c_0))
        else:
            c_n = None
            out_pack, h_n = self.rnn(x_p, h_0)

        # unsort h_n
        h_n = h_n.select(1, x_unsort_idx)
        if only_use_last_hidden_state:
            return h_n
        else:
            # unpack and unsort out
            out, lengths = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out.select(0 if self.batch_first else 1, x_unsort_idx)
            lengths = lengths.numpy()
            lengths = lengths[np.argsort(x_sort_idx)]

            # unsort c_n
            if c_n is not None:
                c_n = c_n.select(1, x_unsort_idx)
                return (out, lengths), (h_n, c_n)
            else:
                return (out, lengths), h_n


class Attn(nn.Module):
    def __init__(self, method, encode_hidden_size, decode_hidden_size):
        super(Attn, self).__init__()
        if method.lower() not in ["dotted", "general", "concat"]:
            raise RuntimeError("Attention methods should be dotted, general or concat but get {}!".format(method))
        if method.lower() == "dotted" and encode_hidden_size != decode_hidden_size:
            raise RuntimeError("In dotted attention, the encode_hidden_size should equal to decode_hidden_size.")

        self.method = method.lower()
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size

        if self.method == "general":
            self.attn = nn.Linear(self.encode_hidden_size, self.decode_hidden_size)
        elif self.method == "concat":
            self.attn = nn.Sequential(
                nn.Linear((self.encode_hidden_size + self.decode_hidden_size),
                          (self.encode_hidden_size + self.decode_hidden_size) / 2),
                nn.Tanh(),
                nn.Linear((self.encode_hidden_size + self.decode_hidden_size) / 2, 1)
            )

    def forward(self, encode_outputs, decode_state):
        # todo: add mask
        """
        :param encode_outputs: (batch, output_length, encode_hidden_size)
        :param decode_state: (batch, decode_hidden_size)
        :return: probs: (batch, output_length), attention results: (batch, encode_hidden_size)
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
        probs_temp = probs.unsqueeze(2)
        results = torch.sum(probs_temp * encode_outputs, 1)
        return probs, results
