import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils import true_randperm
from utils import resize


class Spatial_Attn(nn.Module):
    def __init__(self,word_dim, img_channel, img_hw):
        super(Spatial_Attn,self).__init__()
        self.FC = nn.Linear(word_dim,img_channel)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,feat,word_emb): #b,c,h,w; b,l,768
        b,c,h,w = feat.shape
        feat = feat.view(b,c,-1).permute(0,2,1) #b,c,h,w --> b,c,h*w --> b,h*w,c
        word_emb = self.FC(word_emb).permute(0,2,1) #b,l,768 --> b,l,c --> b,c,l
        att = self.softmax(torch.bmm(feat,word_emb)) #b,h*w,c;b,c,l --> b,h*w,l
        att_feat = torch.bmm(att,word_emb.permute(0,2,1)) #b,h*w,c
        return feat.permute(0,2,1).view(b,c,h,w)


class Channel_Attn(nn.Module):
    def __init__(self,word_dim, img_channel, img_hw):
        super(Channel_Attn,self).__init__()
        self.FC = nn.Linear(word_dim,img_hw*img_hw)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,feat,word_emb): #b,c,h,w; b,l,768
        b,c,h,w = feat.shape
        feat = feat.view(b,c,-1) #b,c,h,w --> b,c,h*w
        word_emb = self.FC(word_emb).permute(0,2,1) #b,l,768 --> b,l,h*w --> b,h*w,l
        att = self.softmax(torch.bmm(feat,word_emb)) #b,c,h*w;b,h*w,l --> b,c,l
        att_feat = torch.bmm(att,word_emb.permute(0,2,1)) #b,c,h*w
        return feat.view(b,c,h,w)


class Decoupling_Attn(nn.Module):
    """ Passive Attention Layer
        learns a fixed attention mask for every channel (c) of feature map (hxw)
    """

    def __init__(self, c, h, w, softmax=True):
        super(Decoupling_Attn, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.attention = nn.Parameter(torch.Tensor(c, h * w).normal_(0, 1))
        self.softmax = softmax
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        if self.softmax:
            return input + self.gamma * input * torch.softmax(self.attention, dim=1).view(self.c, self.h,
                                                                                          self.w).expand_as(input)
        else:
            return input + self.gamma * input * self.attention.view(self.c, self.h, self.w).expand_as(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class UnFlatten(nn.Module):
    def __init__(self, block_size=1):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=True):
    "1x1 convolution with padding"
    return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias))


def conv3x3(in_planes, out_planes, bias=True):
    "3x3 convolution with padding"
    return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=bias))


class Upscale2d(nn.Module):
    def __init__(self, factor):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        s = x.size()
        x = x.view(-1, s[1], s[2], 1, s[3], 1)
        x = x.expand(-1, s[1], s[2], self.factor, s[3], self.factor)
        x = x.contiguous().view(-1, s[1], s[2] * self.factor, s[3] * self.factor)
        return x

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        Upscale2d(factor=2),
        conv3x3(in_planes, out_planes*2),
        nn.BatchNorm2d(out_planes*2),
        GLU(),
        #conv3x3(out_planes, out_planes*2),
        #nn.BatchNorm2d(out_planes*2),
        #GLU(),
        )
    return block

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        #nn.LeakyReLU(0.2),
        #conv3x3(out_planes, out_planes),
        #nn.BatchNorm2d(out_planes),
        nn.AvgPool2d(2),
        nn.LeakyReLU(0.2),
        )
    return block

# Keep the spatial size
def block3x3(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes*2),
        nn.BatchNorm2d(out_planes*2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num*2),
            nn.BatchNorm2d(channel_num*2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out



