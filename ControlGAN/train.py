import os
import numpy as np
from PIL import Image
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

from utils import CaptionImageDataset
from pytorch_transformers import BertModel, BertTokenizer
from ControlGAN_model import Text_bert,G,D

image_size = 256
batch_size = 4
device = torch.device('cuda:0')


pretrained_bert = Text_bert()


transform = transforms.Compose([
    #transforms.Resize(image_size))
    transforms.Resize( int(1.1 * image_size) ),
    #transforms.CenterCrop( int(1.2 * image_size) ),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img_root='data/CUB_200_2011/images'
img_meta_root='data/file_with_bert_cap.pkl'

data = CaptionImageDataset(img_root, img_meta_root, transform=transform)
data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)
dataset = iter(data_loader)

real_image, bert_idx = next(dataset) 
word_emb,mem = pretrained_bert.get_word_emb(bert_idx) #b, 15, 768
sentence = pretrained_bert.get_sentence_emb(mem) #b,256
word_emb,sentence = word_emb.to(device),sentence.to(device)
noise = torch.randn(batch_size, 256).to(device)

controlGAN_G = G().to(device)
controlGAN_D = D().to(device)

fake_image = controlGAN_G(noise,sentence,word_emb)

logits, match, match_perm = controlGAN_D(fake_image, sentence=sentence, word_emb=word_emb,train_perm=True)
print(logits)
print(match)
print(match_perm)
