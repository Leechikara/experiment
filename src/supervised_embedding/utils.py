# coding=utf-8
import numpy as np


def batch_iter(tensor, batch_size, shuffle=False):
    if tensor.shape[0] % batch_size > 0:
        extra_data_num = batch_size - tensor.shape[0] % batch_size
        extra_tensor = np.random.permutation(tensor)[:extra_data_num]
        tensor = np.append(tensor, extra_tensor, axis=0)
    batches_count = tensor.shape[0] // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
        data = tensor[shuffle_indices]
    else:
        data = tensor

    for batch_num in range(batches_count):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        yield data[start_index:end_index]


def neg_sampling_iter(tensor, batch_size, count, seed=None):
    if tensor.shape[0] % batch_size > 0:
        extra_data_num = batch_size - tensor.shape[0] % batch_size
        extra_tensor = np.random.permutation(tensor)[:extra_data_num]
        tensor = np.append(tensor, extra_tensor, axis=0)
    batches_count = tensor.shape[0] // batch_size

    trials = 0
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    data = tensor[shuffle_indices]
    for batch_num in range(batches_count):
        trials += 1
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        if trials > count:
            return
        else:
            yield data[start_index:end_index]
