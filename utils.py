"""
@Author:        禹棋赢
@StartTime:     2021/10/13 20:58
@Filename:      utils.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """
    perform softmax on the last axis by masking elements beyond valid_lens
    :param X: 3D tensor
    :param valid_lens: 1D or 2D tensor
    :return:
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:  # if valid_lens is 1D, each 2D matrix of X has a same valid_len
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
    return F.softmax(X.reshape(shape), dim=-1)


masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
