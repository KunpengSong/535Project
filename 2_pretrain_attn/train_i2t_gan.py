from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import pickle
import os
from copy import deepcopy
from shutil import copy
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils as vutils

from utils import CaptionImageDataset
from pytorch_transformers import BertModel, BertTokenizer
from itertools import chain
from models import G, D, D_with_t, WordAttention, SentenceAttention, ImageAE
from text_models import Text_VAE, Text_Latent_D, TextFromImageG, TextFromImageD

def image_cap_loader(img_root='/media/bingchen/research3/CUB_birds/CUB_200_2011/images'):
    img_meta_root = img_root
    img_meta_root = img_meta_root.replace('images','birds_meta')
    def loader(transform, batch_size=4):

        data = CaptionImageDataset(img_root, img_meta_root, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)
        return data_loader
    return loader


def sample_data(dataloader, image_size=4, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize( int(1.1 * image_size) ),
        #transforms.CenterCrop( int(1.2 * image_size) ),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform, batch_size)

    return loader

import numpy as np
normalization = torch.Tensor([np.log(2 * np.pi)])
def NLL(sample, params):
    """Analytically computes
       E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
       If mu_2, and sigma_2^2 are not provided, defaults to entropy.
    """
    mu = params[:,:,0]
    logsigma = params[:,:,1]
        
    c = normalization.to(mu.device)
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return torch.mean(0.5 * (tmp * tmp + 2 * logsigma + c))


def make_target(word_idcs):
    target = torch.zeros(word_idcs.size(0), 2100).cuda()
    for idx in range(word_idcs.shape[0]):
        target[idx][word_idcs[idx]] = 1
    return target


from random import shuffle
def true_randperm(size, device=torch.device("cuda:0")):
    def unmatched_randperm(size):
        l1 = [i for i in range(size)]
        l2 = []
        for j in range(size):
            deleted = False
            if j in l1:
                deleted = True
                del l1[l1.index(j)]
            shuffle(l1)
            if len(l1) == 0:
                return 0, False
            l2.append(l1[0])
            del l1[0]
            if deleted:
                l1.append(j)
        return l2, True
    flag = False
    while not flag:
        l, flag = unmatched_randperm(size)
    return torch.LongTensor(l).to(device)


def train_image_gan_with_text(net_ig, net_id, opt_ig, opt_id, total_iter, loader, options):
    

    text_g_val = 0
    text_d_val = 0
    text_dt_val = 0
    text_gt_val = 0
    text_dt_mis_val = 0

    log_folder = options.trial_name
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')
    
    log_file_name = os.path.join(log_folder, 'train_image_to_text_log.txt')
    log_file = open(log_file_name, 'w')
    log_file.write('rec, prob, code\n')
    log_file.close()

    copy('train_i2t_gan.py', log_folder+'/train_i2t_gan.py')
    copy('models.py', log_folder+'/models.py')


    data_loader = sample_data(loader, image_size=128, batch_size=options.batch_size)
    dataset = iter(data_loader)

    for i in tqdm(range(options.start_iter, options.total_iter)):

        try:
            real_image, bird_idx, bert_idx = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, bird_idx, bert_idx = next(dataset)


        ### 1. load the data
        b_size = real_image.shape[0]
        real_image = real_image.cuda()
        real_embs = net_t_ae.bert(bert_idx.cuda())[0]
        real_text_latent = net_t_ae.encode(real_embs).detach()

        bird_idx = bird_idx.long().cuda()

        perm = true_randperm(b_size)

        img_feat_16, img_feat_8, img_feat_4 = net_iae.encoder(real_image)
        # 2. Train the Generators

        if i==(options.total_iter//4) and options.checkpoint is None:
            opt_tg.add_param_group({'params': chain(net_tg.word_attn_4.parameters(), 
                                net_tg.word_attn_16.parameters(),
                                net_tg.sentence_attn_4.parameters(),  
                                net_tg.sentence_attn_16.parameters(), 
                                ), 'lr': 0.1*args.lr})


        net_tg.zero_grad()
        
        noise = torch.randn(b_size, 128).cuda()
        
        g_text_latent = net_tg(noise, img_feat_4, img_feat_16)
        g_pred = net_td(g_text_latent)
        g_pred_i = net_tdi(g_text_latent, img_feat_4, img_feat_16)

        loss_g_latent = -g_pred.mean() - g_pred_i.mean()

        loss_total = loss_g_latent
        loss_total.backward()
        
        opt_tg.step()


        text_g_val += g_pred.mean().item()
        text_gt_val += g_pred_i.mean().item()

        ### 3. Train the Discriminators
        if i==(options.total_iter//4) and options.checkpoint is None:
            opt_tdi.add_param_group({'params': chain(
                                net_tdi.sentence_attn_4.parameters(), 
                                net_tdi.sentence_attn_16.parameters(),
                                ), 'lr': 0.1*args.lr})
        ### 3.1 train the image-only discriminator
        net_id.zero_grad()

        real_predict = net_td(real_text_latent)
        fake_predict = net_id(g_text_latent.detach())
        
        loss_disc = F.relu(1-real_predict).mean() + F.relu(1+fake_predict).mean() 
        loss_disc.backward()
        opt_td.step()
        text_d_val += real_predict.mean().item()

        ### 3.2 train the image-text discriminator
        net_tdi.zero_grad()

        real_predict = net_tdi(real_text_latent, img_feat_4, img_feat_16)
        fake_predict = net_tdi(g_text_latent.detach(), img_feat_4, img_feat_16)
        mismatch_predict = net_tdi(real_text_latent, img_feat_4[perm], img_feat_16[perm])

        loss_disc = F.relu(1-real_predict).mean() + \
                    F.relu(1+fake_predict).mean() + \
                    F.relu(1+mismatch_predict).mean()

        loss_disc.backward()
        opt_tdi.step()
        
        text_dt_val += real_predict.mean().item()
        text_dt_mis_val += mismatch_predict.mean().item()

        ### 4. Logging 
        if (i + 1) % 2000 == 0 or i==0:
            with torch.no_grad():            
                vutils.save_image(real_image.detach().add(1).mul(0.5), f'{log_folder}/sample/r_img_{str(i + 1).zfill(6)}.jpg')
                real_texts = net_t_ae.generate(real_text_latent)
                g_texts = net_t_ae.generate(g_text_latent)
                f = open(f'{log_folder}/sample/g_real_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for cap in real_texts+g_texts:
                    f.write(cap+'\n')
                f.close()    

        if (i+1) % 5000 == 0 or i==0:
            torch.save({'tg':net_tg.state_dict(), 'td':net_td.state_dict(), 'tdi':net_tdi.state_dict()}, f'{log_folder}/checkpoint/image_to_text_memory_{str(i + 1).zfill(6)}_model.pth')
            torch.save({'tg':opt_tg.state_dict(), 'td':opt_td.state_dict(), 'tdi':opt_tdi.state_dict()}, f'{log_folder}/checkpoint/image_to_text_memory_{str(i + 1).zfill(6)}_opt.pth')
            
        interval = 100
        if (i+1)%interval == 0:
            
            state_msg = (f'txt_g_val: {text_g_val/(interval):.3f};   txt_d_val: {text_d_val/interval:.3f};   \n'
                        f'txt_gt_val: {text_gt_val/(interval):.3f};   txt_dt_val: {text_dt_val/interval:.3f};  txt_dt_mis: {text_dt_mis_val/interval:.3f} \n')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f,%.5f\n"%\
                (text_g_val/(interval), text_d_val/interval, text_dt_val/interval)
            log_file.write(new_line)
            log_file.close()

            text_g_val = 0
            text_d_val = 0
            text_gt_val = 0
            text_dt_val = 0
            text_dt_mis_val = 0
            print(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Text Generation Together')

    parser.add_argument('--path', type=str, default='../../../research3/CUB_birds/CUB_200_2011/images', help='path of specified dataset')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--trial_name', default='trial_i2t_gan_with_pre-trained_sw', type=str, help='name of the trial')
    parser.add_argument('--total_iter', default=300000, type=int, help='iterations')
    parser.add_argument('--start_iter', default=0, type=int, help='start iterations')
    parser.add_argument('--im_size', default=128, type=int, help='initial image size')
    parser.add_argument('--batch_size', default=8, type=int, help='initial image size')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load pre-trained model')
    parser.add_argument('--channel', type=int, default=128, help='channel number in models')
    parser.add_argument('--ae_path', type=str, default=None, help='path to load pre-trained text Autoencoder model')
    
    args = parser.parse_args()

    img_meta_root = str(args.path).replace('images','birds_meta')
    
    # creating Text model
    pre_trained_path = './trial_it_attn/checkpoint/it_ae_160000_model.pth'
    checkpoint = torch.load(pre_trained_path)
    net_t_ae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=img_meta_root).cuda()
    net_t_ae.load_state_dict(checkpoint['t'])
    net_t_ae.eval()
    for p in net_t_ae.parameters():
        p.requires_grad = False

    net_iae = ImageAE(channel=256).cuda()
    net_iae.load_state_dict(checkpoint['i'])
    net_iae.eval()
    for p in net_iae.parameters():
        p.requires_grad = False

    net_tg = TextFromImageG()
    net_tg.cuda()

    net_tg.sentence_attn_4.load_state_dict(checkpoint['sa4'])
    net_tg.sentence_attn_16.load_state_dict(checkpoint['sa16'])
    net_tg.word_attn_4.load_state_dict(checkpoint['wa4'])
    net_tg.word_attn_16.load_state_dict(checkpoint['wa16'])


    net_td = Text_Latent_D()
    net_td.cuda()

    net_tdi = TextFromImageD()
    net_tdi.cuda()

    net_tdi.sentence_attn_4.load_state_dict(checkpoint['sa4'])
    net_tdi.sentence_attn_16.load_state_dict(checkpoint['sa16'])

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net_tg.load_state_dict(checkpoint['tg'])
        net_td.load_state_dict(checkpoint['td'])
        net_tdi.load_state_dict(checkpoint['tdi'])

    opt_tg = optim.Adam( chain( net_tg.text_values.parameters(),
                                net_tg.final.parameters(),
                                net_tg.sentence_receiver.parameters(),
                                net_tg.sentence_receiver.parameters(),
                                ), lr=args.lr, betas=(0.5, 0.99))
    

    opt_td = optim.Adam( net_td.parameters(), lr=args.lr, betas=(0.5, 0.99))
    
    opt_tdi = optim.Adam( chain(
        net_tdi.text_values.parameters(), net_tdi.final.parameters()), lr=args.lr, betas=(0.5, 0.99))
    
    

    if args.checkpoint is not None:
        opt_tg.add_param_group({'params': chain(net_tg.word_attn_4.parameters(), 
                                net_tg.word_attn_16.parameters(),
                                net_tg.sentence_attn_4.parameters(),  
                                net_tg.sentence_attn_16.parameters(), 
                                ), 'lr': 0.1*args.lr})
        opt_tdi.add_param_group({'params': chain(
                                net_tdi.sentence_attn_4.parameters(), 
                                net_tdi.sentence_attn_16.parameters(),
                                ), 'lr': 0.1*args.lr})

        checkpoint = torch.load(args.checkpoint.replace('model.pth', 'opt.pth'))
        opt_tg.load_state_dict(checkpoint['tg'])
        opt_td.load_state_dict(checkpoint['td'])
        opt_tdi.load_state_dict(checkpoint['tdi'])

    print(args.path)
    loader = image_cap_loader(args.path)

    total_iter = args.total_iter
    train_image_gan_with_text(net_tg, net_td, opt_tg, opt_td, total_iter, loader, args)