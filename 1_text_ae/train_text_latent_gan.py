from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import pickle
import os
from shutil import copy
import matplotlib.pyplot as plt

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


def train_text_gan(net_tg, net_td, opt_tg, opt_td, total_iter, loader, options):
    data_loader = sample_data(loader, image_size=options.im_size, batch_size=options.batch_size)
    dataset = iter(data_loader)

    text_g_val = 0
    text_d_val = 0

    log_folder = options.trial_name
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')
    
    log_file_name = os.path.join(log_folder, 'train_txt_gan_log.txt')
    log_file = open(log_file_name, 'w')
    log_file.write('g, d, kl\n')
    log_file.close()

    copy('train_text_latent_gan.py', log_folder+'/train_text_latent_gan.py')
    copy('models.py', log_folder+'/models.py')
  

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
        net_tg.zero_grad()

        real_embs = net_t_ae.bert(bert_idx)[0]
        real_text_latent = net_t_ae.encode(real_embs)

        addon_weight = max(0, 1 - 0.0001*i)
        real_text_latent = (1-addon_weight)*real_text_latent + addon_weight * torch.randn_like(real_text_latent)
        
        noise = torch.randn_like(real_text_latent)
        g_text_latent = net_tg(noise)
        
        loss_g_latent = - net_td(g_text_latent).mean()
                
        loss_total = loss_g_latent 
        loss_total.backward()
        opt_tg.step()

        text_g_val += -loss_g_latent.item()

        ### 3. Train the Discriminators
        net_td.zero_grad()

        real_predict_latent = net_td(real_text_latent.detach())

        fake_predict_latent = net_td(g_text_latent.detach())
        
        loss_disc = F.relu(1-real_predict_latent).mean() + F.relu(1+fake_predict_latent).mean() 
        loss_disc.backward()
        opt_td.step()

        text_d_val += real_predict_latent.mean().item()

        ### 4. Logging 
        if (i + 1) % 2000 == 0 or i==0:
            with torch.no_grad():            
                r = real_text_latent[:,:2].detach().cpu().data.numpy()
                g = g_text_latent[:,:2].detach().cpu().data.numpy()
                plt.plot(r[:,0], r[:,1], 'ro')
                plt.plot(g[:,0], g[:,1], 'bo')
                plt.savefig(f'{log_folder}/sample/text_gan_fig_{str(i + 1).zfill(6)}_01.pdf')

                r = real_text_latent[:,2:4].detach().cpu().data.numpy()
                g = g_text_latent[:,2:4].detach().cpu().data.numpy()
                plt.plot(r[:,0], r[:,1], 'ro')
                plt.plot(g[:,0], g[:,1], 'bo')
                plt.savefig(f'{log_folder}/sample/text_gan_fig_{str(i + 1).zfill(6)}_23.pdf')

                r = real_text_latent[:,4:6].detach().cpu().data.numpy()
                g = g_text_latent[:,4:6].detach().cpu().data.numpy()
                plt.plot(r[:,0], r[:,1], 'ro')
                plt.plot(g[:,0], g[:,1], 'bo')
                plt.savefig(f'{log_folder}/sample/text_gan_fig_{str(i + 1).zfill(6)}_45.pdf')


                g_captions_idx = net_t_ae.generate(g_text_latent)
                file = open(f'{log_folder}/sample/text_gan_g_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for g_cap in g_captions_idx:
                    file.write(g_cap+'\n')
                file.close()
    

        if (i+1) % 10000 == 0 or i==0:
            try:
                torch.save({'tg':net_tg.state_dict(), 'td':net_td.state_dict()}, f'{log_folder}/checkpoint/text_gan_{str(i + 1).zfill(6)}_model.pth')
                torch.save({'tg':opt_tg.state_dict(), 'td':opt_td.state_dict()}, f'{log_folder}/checkpoint/text_gan_{str(i + 1).zfill(6)}_opt.pth')
            except:
                pass

        if (i+1)%200 == 0:
            
            state_msg = (f' Text_g: {text_g_val/(200):.3f};   Text_d: {text_d_val/200:.3f};\n')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n"%\
                (text_g_val/(200), text_g_val/200)
            log_file.write(new_line)
            log_file.close()

            text_g_val = 0
            text_d_val = 0
            print(state_msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Text Generation Together')

    parser.add_argument('--path', type=str, default='../../../research3/CUB_birds/CUB_200_2011/images', help='path of specified dataset')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--trial_name', default='trial_1', type=str, help='name of the trial')
    parser.add_argument('--total_iter', default=100000, type=int, help='iterations')
    parser.add_argument('--im_size', default=64, type=int, help='initial image size')
    parser.add_argument('--batch_size', default=32, type=int, help='initial image size')
    parser.add_argument('--ae_path', type=str, default=None, help='path to load pre-trained text Autoencoder model')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load pre-trained model')
    parser.add_argument('--channel', type=int, default=128, help='channel number in models')

    args = parser.parse_args()

    img_meta_root = str(args.path).replace('images','birds_meta')
    # creating Text model
    net_t_ae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=img_meta_root).cuda()
    net_t_ae.load_state_dict(torch.load(args.ae_path))
    net_t_ae.eval()
    for p in net_t_ae.parameters():
        p.requires_grad = False

    net_tg = Text_Latent_G(latent=128, noise=128).cuda()
    net_td = Text_Latent_D(latent=128).cuda()

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net_tg.load_state_dict(checkpoint['tg'])
        net_td.load_state_dict(checkpoint['td'])
        
    opt_tg = optim.Adam( net_tg.parameters(), lr=args.lr, betas=(0.1, 0.99))
    opt_td = optim.Adam( net_td.parameters(), lr=args.lr, betas=(0.1, 0.99))

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint.replace('model.pth', 'opt.pth'))
        opt_tg.load_state_dict(checkpoint['tg'])
        opt_td.load_state_dict(checkpoint['td'])

    print(args.path)
    loader = image_cap_loader(args.path)


    total_iter = args.total_iter
    train_text_gan(net_tg, net_td, opt_tg, opt_td, total_iter, loader, args)