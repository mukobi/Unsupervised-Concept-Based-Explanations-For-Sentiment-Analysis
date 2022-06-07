"""
Abstraction-aggregation attention layers.
Copied and modified from https://github.com/tshi04/ACCE/blob/master/LeafNATS/modules/attention
"""

import torch

class AttentionConcepts(torch.nn.Module):
    def __init__(self, input_size, n_concepts):
        """
        Multi-headed self-attention: abstraction-attention (abs).
        https://github.com/tshi04/ACCE/blob/master/LeafNATS/modules/attention/attention_concepts.py
        """
        super().__init__()
        self.n_concepts = n_concepts

        self.ff = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, 1, bias=False)
             for k in range(n_concepts)])

    def forward(self, input_, mask=None):
        """
        input vector: input_
        output:
            attn_weights: attention weights
            attn_ctx_vec: context vector
        """
        input_ = input_.last_hidden_state
        batch_size = input_.size(0)

        attn_weight = []
        attn_ctx_vec = []
        for k in range(self.n_concepts):
            attn_ = self.ff[k](input_).squeeze(2)
            if mask is not None:
                attn_ = attn_.masked_fill(mask == 0, -1e9)
            attn_ = torch.softmax(attn_, dim=1)
            ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)
            attn_weight.append(attn_)
            attn_ctx_vec.append(ctx_vec)

        attn_weight = torch.cat(attn_weight, 0).view(
            self.n_concepts, batch_size, -1)
        attn_weight = attn_weight.transpose(0, 1)
        attn_ctx_vec = torch.cat(attn_ctx_vec, 0).view(
            self.n_concepts, batch_size, -1)
        attn_ctx_vec = attn_ctx_vec.transpose(0, 1)

        return attn_weight, attn_ctx_vec


class AttentionSelf(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout_rate=None
    ):
        """
        Single-headed self-attention: aggregation attention (agg).
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, 1, bias=False)
        if dropout_rate is not None:
            self.model_drop = torch.nn.Dropout(dropout_rate)

    def forward(self, input_, mask=None):
        """
        input vector: input_
        output:
            attn_: attention weights
            ctx_vec: context vector
        """
        attn_ = torch.tanh(self.ff1(input_))
        attn_ = self.ff2(attn_).squeeze(2)
        if mask is not None:
            attn_ = attn_.masked_fill(mask == 0, -1e9)
        # dropout method 1.
        if self.dropout_rate is not None:
            # TODO variables are depr https://pytorch.org/docs/stable/autograd.html#variable-deprecated
            drop_mask = torch.ones(attn_.size()).to(input_.device)
            drop_mask = self.model_drop(drop_mask)
            attn_ = attn_.masked_fill(drop_mask == 0, -1e9)

        attn_ = torch.softmax(attn_, dim=1)
        # dropout method 2.
        if self.dropout_rate is not None:
            attn_ = self.model_drop(attn_)
        ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)

        return attn_, ctx_vec
