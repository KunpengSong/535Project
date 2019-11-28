# -*- coding:utf8 -*-


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class WordLevelDiscriminator(nn.Module):
    def __init__(self, img_feat_dim, word_feat_dim):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.word_feat_dim = word_feat_dim
        self.F_prime = nn.Linear(img_feat_dim, word_feat_dim)

    def forward(self, img_feat, word_feat, word_num):
        #batch channel height width
        b, c, h, w = img_feat.shape
        #d = self.word_feat_dim
        #batch feature_dim max_word_num
        b, d, l = word_feat.shape

        img_feat = self.F_prime(
            img_feat.permute(0, 2, 3, 1).contiguous().view(-1, c)
        ).view(b, h * w, d).permute(0, 2, 1) # B D H*W

        # B L H*W
        mm1 = torch.bmm(word_feat.transpose(1, 2), img_feat).softmax(dim=2)

        # B D L
        mm2 = torch.bmm(img_feat, mm1.transpose(2, 1))

        w_wlsa = self.word_level_self_attention(word_feat, word_num.type(torch.float))
        # B D L
        b_bar = mm2 * w_wlsa

        l_corre = cosine_similarity(word_feat, b_bar).sigmoid()\
            .flatten(start_dim=1).sum(dim=1, keepdim=True)
        return l_corre

    def word_level_self_attention(self, word_feat, word_num):
        mean_word_feat = word_feat.sum(dim=1, keepdim=True) / word_num.unsqueeze(-1)
        attn_txt_exp = (mean_word_feat * word_feat).exp()
        attn_txt = attn_txt_exp / attn_txt_exp.sum(0, keepdim=True)
        return attn_txt


if __name__ == "__main__":
    img_feature = torch.randn(10, 3, 24, 24)
    word_feature = torch.randn(10, 16, 50)
    word_num = torch.randint(5, 10, (10, 1))

    wd = WordLevelDiscriminator(3, 16)
    ret = wd(img_feature, word_feature, word_num)
    print(ret, ret.shape)
    pass
