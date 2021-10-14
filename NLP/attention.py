"""
@Author:        禹棋赢
@StartTime:     2021/10/14 11:49
@Filename:      attention.py
"""

import torch
import math
from torch import nn

from utils import masked_softmax


class DotProductAttention(nn.Module):
    """Scaled Dot Production"""

    def __init__(self, drp):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(drp)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        :param queries: (bsz, num. of queries, hidden_size)
        :param keys: (bsz, num. of key-value pairs, hidden_size)
        :param values: (bsz, num. of key-value pairs, value_dim)
        :param valid_lens: (bsz,) or (bsz, query_num), indicate each query consider how many key-value pairs
        :return:
        """
        d = queries.shape[-1]
        atten_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # (bsz, query_num, key-value num)
        atten_weights = masked_softmax(atten_scores, valid_lens)
        # add drpout on attention weights
        return torch.bmm(self.dropout(atten_weights), values)  # (bsz, query_num, value_dim)
