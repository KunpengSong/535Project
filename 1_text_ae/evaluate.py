from models import Text_VAE, Text_Latent_G
from utils import CaptionImageDataset, transform
from pytorch_transformers import BertModel, BertTokenizer
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchnlp.metrics import get_moses_multi_bleu
from random import shuffle
import numpy as np
from bert_score import score as b_score

from tqdm import tqdm

data_root = '/media/bingchen/research3/CUB_birds/CUB_200_2011/images'
data_meta_root = data_root.replace('images','birds_meta')

net_bert = BertModel.from_pretrained('bert-base-uncased')
net_t_ae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=data_meta_root)
net_t_ae.load_state_dict(torch.load('./trial_1/checkpoint/text_ae_200000_model.pth'))
net_t_ae.cuda()
net_t_ae.eval()

net_g = Text_Latent_G()

data = CaptionImageDataset(data_root, data_meta_root, transform=transform)
data_loader = DataLoader(data, shuffle=True, batch_size=32, num_workers=4)

bird2bert = pickle.load((open(data_meta_root+'/bird2bert.pkl', 'rb')))
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



real_texts = []
dl = iter(data_loader)
for i in range(100):
    try:
        data = next(dl)
    except:
        dl = iter(data_loader)
    _, bird_idx, bert_idx = data
    for bert_idc in bert_idx:
        bird_cap = ''.join(list(map( lambda i: bertTokenizer.decode(i).replace(' ','')+' ', [bidx.item() for bidx in bert_idc] )))   
        real_texts.append(bird_cap)
real_whole_texts = ''
for t in real_texts:
    real_whole_texts += t


all_scores_bert = []
all_scores_bleu = []
all_stds_bleu = []

for iteration in range(1,11):
    checkpoint_path = f'trial_1/checkpoint/text_gan_{str(iteration*10000).zfill(6)}_model.pth'
    checkpoint = torch.load(checkpoint_path)
    net_g.load_state_dict(checkpoint['tg'])

    net_g.cuda()
    net_g.eval()


    fake_texts = []
    for i in range(100):
        noise = torch.randn(32, 128).cuda()
        g_text_latent = net_g(noise)
        g_captions = net_t_ae.generate(g_text_latent)
        
        fake_texts += g_captions

    #p,r,f = b_score(fake_texts, real_texts, bert="bert-base-uncased", verbose=True)
    #print(f.mean().item())
    #all_scores_bert.append(f.mean().item())

    fake_whole_texts = ''
    for t in fake_texts:
        fake_whole_texts += t
    score = get_moses_multi_bleu([fake_whole_texts], [real_whole_texts], lowercase=True)
    all_scores_bleu.append(score)

print(all_scores_bert, all_scores_bleu)