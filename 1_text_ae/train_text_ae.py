from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import pickle
import os
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
from itertools import chain
from models import Text_Latent_G, Text_Latent_D, Text_VAE


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
        transforms.Resize(image_size+int(image_size*0.1)+1),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform, batch_size)

    return loader


def train_text_ae(net_t_ae, opt_tae, total_iter, loader, options):
    data_loader = sample_data(loader, image_size=options.im_size, batch_size=options.batch_size)
    dataset = iter(data_loader)

    text_emb_recons_val = 0
    text_idx_recons_val = 0
    text_latent_kl_val = 0

    log_folder = options.trial_name
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')
    
    log_file_name = os.path.join(log_folder, 'train_txt_ae_log.txt')
    log_file = open(log_file_name, 'w')
    log_file.write('emb_recons, idx_recons, kl\n')
    log_file.close()

    copy('train_text_ae.py', log_folder+'/train_text_ae.py')
    copy('models.py', log_folder+'/models.py')

    ### select one batch for logging purpose
    real_image, bird_idx, bert_idx = next(dataset)
    fixed_real_txt = bert_idx.cuda()

    file = open(f'{log_folder}/sample/recon_txt_gt.txt', 'w')
    for bert_idc in fixed_real_txt.clone():
        bird_cap = ''.join(list(map( lambda i: net_t_ae.bertTokenizer.decode(i).replace(' ','')+' ', [bidx.item() for bidx in bert_idc] )))   
        file.write(bird_cap+'\n')
    file.close()
    vutils.save_image(real_image.clone(), f'{log_folder}/sample/recon_img_gt.jpg', normalize=True, range=(-1,1))
    ### select one batch for logging purpose    

    for i in tqdm(range(options.total_iter)):
        try:
            real_image, bird_idx, bert_idx = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, bird_idx, bert_idx = next(dataset)


        ### 1. load the data
        bird_idx = bird_idx.cuda()
        bert_idx = bert_idx.cuda()
            
        # 2. Train the Generators
        net_t_ae.zero_grad()

        real_embs = net_t_ae.bert(bert_idx)[0]
        real_text_latent = net_t_ae.encode(real_embs)
        recon_text_logits, recon_embs = net_t_ae.decode(real_text_latent)

        loss_txt_idx_recon = F.cross_entropy(recon_text_logits, bird_idx.view(-1))
        loss_txt_emb_recon = F.l1_loss(recon_embs, real_embs)
        loss_txt_latent_regularizer = F.l1_loss(real_text_latent.mean(), torch.tensor(0.0).cuda()) +\
                                        F.l1_loss(real_text_latent.std(), torch.tensor(1.0).cuda())

        loss_total = loss_txt_idx_recon + loss_txt_emb_recon + loss_txt_latent_regularizer
        loss_total.backward()
        opt_tae.step() 

        text_emb_recons_val += loss_txt_emb_recon.item()
        text_idx_recons_val += loss_txt_idx_recon.item()
        text_latent_kl_val += loss_txt_latent_regularizer.item()

        ### 4. Logging 
        if (i + 1) % 2000 == 0 or i==0:
            with torch.no_grad():            
                embs = net_t_ae.bert(fixed_real_txt)[0]
                text_latent = net_t_ae.encode(embs)
                g_captions_idx = net_t_ae.generate(text_latent)
                file = open(f'{log_folder}/sample/recons_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for g_cap in g_captions_idx:
                    file.write(g_cap+'\n')
                file.close()
    

        if (i+1) % 10000 == 0 or i==0:
            torch.save(net_t_ae.state_dict(), f'{log_folder}/checkpoint/text_ae_{str(i + 1).zfill(6)}_model.pth')
            torch.save(opt_tae.state_dict(), f'{log_folder}/checkpoint/text_ae_{str(i + 1).zfill(6)}_opt.pth')
            

        if (i+1)%200 == 0:
            
            state_msg = (f' Text_idx_ce: {text_idx_recons_val/(200):.3f};   Text_emb_ce: {text_emb_recons_val/200:.3f};\n'
                f' text_latent_kl_val: {text_latent_kl_val/(200):.3f}; ')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f,%.5f\n"%\
                (text_emb_recons_val/(200), text_idx_recons_val/200, text_latent_kl_val/(200))
            log_file.write(new_line)
            log_file.close()

            text_emb_recons_val = 0
            text_idx_recons_val = 0
            text_latent_kl_val = 0
            print(state_msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Text Generation Together')

    parser.add_argument('--path', type=str, default='../../../research3/CUB_birds/CUB_200_2011/images', help='path of specified dataset')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--trial_name', default='trial_1', type=str, help='name of the trial')
    parser.add_argument('--total_iter', default=200000, type=int, help='iterations')
    parser.add_argument('--im_size', default=64, type=int, help='initial image size')
    parser.add_argument('--batch_size', default=32, type=int, help='initial image size')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load pre-trained model')
    parser.add_argument('--channel', type=int, default=128, help='channel number in models')

    args = parser.parse_args()

    img_meta_root = str(args.path).replace('images','birds_meta')
    # creating Text model
    net_t_ae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=img_meta_root).cuda()

    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net_t_ae.load_state_dict(checkpoint)
        
    opt_tae = optim.Adam( chain(net_t_ae.encoder.parameters(),\
                                    net_t_ae.decoder.parameters(),\
                                        net_t_ae.decoder_fc.parameters()) , lr=args.lr, betas=(0.0, 0.99))
    opt_tae.add_param_group({'params': net_t_ae.bert.parameters(), 'lr': args.lr * 0.1})

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint.replace('model.pth', 'opt.pth'))
        opt_tae.load_state_dict(checkpoint)

    print(args.path)
    loader = image_cap_loader(args.path)


    total_iter = args.total_iter
    train_text_ae(net_t_ae, opt_tae, total_iter, loader, args)