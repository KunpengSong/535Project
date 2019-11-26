import torch
from torch import nn
from modules import UnFlatten, GLU, upBlock, SpatialAttn, ChannelAttn, Decoupling_Attn
from modules import ResBlock, conv1x1, conv3x3, block3x3, weights_init
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from utils import true_randperm
from modules import downBlock


class Attn(nn.Module):
    def __init__(self, word_emb=512, img_hw=32, img_channel=128):
        super().__init__()

        self.decup_attn = Decoupling_Attn(img_channel, img_hw, img_hw)
        self.spatial_attn = SpatialAttn(word_emb, img_hw, img_channel)
        self.conv = block3x3(img_channel*2, img_channel)

    def forward(self, img_feat, word_emb):
        img_feat = self.decup_attn(img_feat)
        feat = torch.cat([img_feat, self.spatial_attn(img_feat, word_emb)], dim=1)
        return self.conv(feat)
