# -*- coding:utf8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import pickle
from copy import deepcopy
from shutil import copy
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils as vutils
from torch import optim
from utils import *

from pytorch_transformers import BertModel, BertTokenizer, BertConfig
from ControlGAN_model import Text_bert,G,D
#from ControlGAN_model import Text_bert_LSTM


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


SMOOTH_GAMMA1 = 4.0
SMOOTH_GAMMA2 = 5.0
SMOOTH_GAMMA3 = 10.0


class Text_bert_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig()
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.ninput = 768
        self.nlayers = 1
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.nhidden = 256 // self.num_directions
        self.drop_prob = 0.5
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)

    def get_word_emb(self,words): #tensor: b,15
        embs = self.bert(words)
        word_embs = embs[0] #b,15,WORD_DIM
        mem = embs[2] #tuple: length=13, each: b,15,WORD_DIM, last layer same as word_embs
        return word_embs,mem

    def get_sentence_emb(self,mem): # use [4,8,10,12] from [0,1,...,12]
        mem4,mem8,mem10,mem12 = mem[4],mem[8],mem[10],mem[12]

        sentence = torch.cat([mem4,mem8,mem10,mem12],dim=-2).permute(0,2,1) #b,15,WORD_DIM --> b,60,WORD_DIM --> b,WORD_DIM,60
        #sentence = nn.MaxPool1d(60)(sentence).squeeze(-1) #b,WORD_DIM,60 --> b,WORD_DIM,1 --> b,WORD_DIM
        sentence = torch.nn.functional.adaptive_avg_pool2d(sentence,(256,1)).squeeze(-1) #b,WORD_DIM,60 --> b,256,1 --> b,256
        return sentence

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * self.num_directions,
                                    batch_size, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions,
                                    batch_size, self.nhidden).zero_()))

    def get_hiddens(self, words, hidden, num_words_each_sample=None):
        embs = self.bert(words)

        if num_words_each_sample is None:
            _num_words_each_sample = [words.shape[1]] * words.shape[0]
        else:
            _num_words_each_sample = num_words_each_sample.data.tolist()

        emb = pack_padded_sequence(embs[0], _num_words_each_sample, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]

        words_emb = output#.transpose(1, 2)
        sent_emb = hidden[0].transpose(0, 1).contiguous().view(
            -1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef, train_flag=False):
        super(CNN_ENCODER, self).__init__()
        if train_flag:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code




def build_models(dataset_n_words, batch_size, num_words, embedding_dim, model_file_path, cuda=True):
    if model_file_path.endswith('.pth'):
        text_encoder_model_path = model_file_path
    else:
        text_encoder_model_path = os.path.join(model_file_path, 'text_encoder200.pth')
    # build model ############################################################
    text_encoder = Text_bert_LSTM(dataset_n_words, num_words, nhidden=embedding_dim)
    image_encoder = CNN_ENCODER(embedding_dim)
    labels = Variable(torch.LongTensor(range(batch_size)))
    if text_encoder_model_path != '':
        state_dict = torch.load(text_encoder_model_path)
        text_encoder.load_state_dict(state_dict)
        print('Load ', text_encoder_model_path)
        #
        name = text_encoder_model_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)
    if cuda:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels


def sent_loss(cnn_code, rnn_code, labels, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    batch_size = rnn_code.size(0)
    masks = []
    #if class_ids is not None:
    #    for i in range(batch_size):
    #        mask = (class_ids == class_ids[i]).astype(np.uint8)
    #        mask[i] = 0
    #        masks.append(mask.reshape((1, -1)))
    #    masks = np.concatenate(masks, 0)
    #    # masks: batch_size x batch_size
    #    masks = torch.ByteTensor(masks)
    #    if cfg.CUDA:
    #        masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * SMOOTH_GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    #if class_ids is not None:
    #    scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def words_loss(img_features, words_emb, labels, cap_lens):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    batch_size = words_emb.size(0)
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        #if class_ids is not None:
        #    mask = (class_ids == class_ids[i]).astype(np.uint8)
        #    mask[i] = 0
        #    masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, SMOOTH_GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(SMOOTH_GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    #if class_ids is not None:
    #    masks = np.concatenate(masks, 0)
    #    # masks: batch_size x batch_size
    #    masks = torch.ByteTensor(masks)
    #    if cfg.CUDA:
    #        masks = masks.cuda()

    similarities = similarities * SMOOTH_GAMMA3
    #if class_ids is not None:
    #    similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    #attn = nn.Softmax()(attn)  # Eq. (8)
    attn = attn.softmax(dim=1)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    #attn = nn.Softmax()(attn)
    attn = attn.softmax(dim=1)  # Eq. (8)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def get_dasam_loss(
        words_features, sent_code, words_emb, sent_emb, labels, cap_lens):
    w_loss0, w_loss1, attn = words_loss(words_features, words_emb.transpose(1, 2), labels, cap_lens)

    s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels)
    loss = w_loss0 + w_loss1 + s_loss0 + s_loss1
    return loss


def train_damsm():
    torch.backends.cudnn.benchmark = True

    image_size = 256
    batch_size = 4
    device = torch.device('cuda:3')

    log_interval = 100
    nepoch = 60
    grad_clip = 0.25
    data_name = 'bird'
    log_folder = get_path('%s_535'%(data_name))
    start_epoch = 0
    embedding_dim = 256

    pretrained_bert = Text_bert_LSTM().to(device)

    image_encoder = CNN_ENCODER(embedding_dim).to(device)
    labels = Variable(torch.LongTensor(range(batch_size))).to(device)

    img_root='/freespace/local/ws383/AttnGAN/data/birds/CUB_200_2011/images'
    img_meta_root='./file_with_bert_cap.pkl'

    data = CaptionImageDataset(img_root, img_meta_root, transform=trans_maker(image_size, resize=True))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)

    losses_gp = AverageMeter()
    losses_g_img = AverageMeter()
    losses_g_match = AverageMeter()
    losses_d_img = AverageMeter()
    losses_d_match = AverageMeter()
    losses_d_mismatch = AverageMeter()

    fixed_inp = None
    fixed_wordidx = None

    para = list()
    for v in pretrained_bert.parameters():
        if v.requires_grad:
            para.append(v)
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    opt_damsm = optim.Adam(para, lr=1e-4, betas=(0.5,0.999))

    for epoch in range(start_epoch, nepoch):

        for batch_idx, (real_image, word_idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            real_image = real_image.to(device)
            word_idx = word_idx.to(device)
            cap_lens = (word_idx != 2203).sum(dim=1)
            sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
            real_image = real_image[sorted_cap_indices]
            word_idx = word_idx[sorted_cap_indices]

            batch_size = word_idx.shape[0]

            hidden = pretrained_bert.init_hidden(batch_size)
            word_emb, sentence = pretrained_bert.get_hiddens(word_idx, hidden, sorted_cap_lens)

            words_features, sent_code = image_encoder(real_image)

            pretrained_bert.zero_grad()
            image_encoder.zero_grad()
            loss = get_dasam_loss(
                words_features, sent_code, word_emb, sentence,
                labels, cap_lens)

            loss.backward()
            opt_damsm.step()
            print("\rLoss: %s" % loss.data.item())

        if epoch % 10 == 0:
            torch.save( {'text_encoder': pretrained_bert.state_dict(),
                         'image_encoder': image_encoder.state_dict(),
                         'optim': opt_damsm.state_dict()}, '%s/checkpoint/damsm_%d.pth'%(log_folder ,epoch))


if __name__ == "__main__":
    train_damsm()

