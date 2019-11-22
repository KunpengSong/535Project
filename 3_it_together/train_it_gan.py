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
from models import G, D, D_with_t, WordAttention, SentenceAttention
from text_models import Text_VAE, Text_fuse_G, Text_fuse_D, Text_Latent_D, Text_Latent_G

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


def train_image_text_gan(net_ig, net_id, net_tg, net_td, net_idt, total_iter, loader, options):
    

    image_g_val = 0
    image_d_val = 0
    image_dt_val = 0
    image_gt_val = 0
    image_dt_mis_val = 0
    text_d_val = 0
    text_g_val = 0


    log_folder = options.trial_name
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')
    
    log_file_name = os.path.join(log_folder, 'train_image_to_text_log.txt')
    log_file = open(log_file_name, 'w')
    log_file.write('rec, prob, code\n')
    log_file.close()

    copy('train_it_gan.py', log_folder+'/train_it_gan.py')
    copy('models.py', log_folder+'/models.py')
    copy('text_models.py', log_folder+'/text_models.py')


    data_loader = sample_data(loader, image_size=64, batch_size=options.batch_size)
    dataset = iter(data_loader)

    step = 0
    for i in tqdm(range(options.start_iter, options.total_iter)):
        if i==options.total_iter//2:
            data_loader = sample_data(loader, image_size=128, batch_size=options.batch_size)
            dataset = iter(data_loader)
            step = 1
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

        
        # the noise-annealing trick: add a noise to the real data and gradually removing it during training
        # make it easier for G to learn and make D less powerful at the very beginning
        addon_weight = max(0, 1 - 0.0001*i)
        real_text_latent_with_noise = (1-addon_weight)*real_text_latent + addon_weight * torch.randn_like(real_text_latent)
        
        bird_idx = bird_idx.long().cuda()

        perm = true_randperm(b_size)

        ### 1. prepare the generated data
        noise_i = torch.randn(b_size, 128).cuda()
        noise_t = noise_i #torch.randn(b_size, 128).cuda()

        g_image_from_real_text = net_ig(noise_i, real_text_latent, real_embs, step)

        g_text_latent = net_tg(noise_t)
        g_embs = net_t_ae.decoder(g_text_latent.unsqueeze(-1).unsqueeze(-1)).view(-1, 15, 768)
        g_image_from_g_text = net_ig(noise_i, g_text_latent, g_embs, step)

        ### 2. Train the Discriminators
        if i==(options.total_iter//4):
            opt_idt.add_param_group({'params': chain(net_idt.d_64.word_attn_4.parameters(), 
                                net_idt.d_64.word_attn_16.parameters(),
                                net_idt.d_64.sentence_attn_4.parameters(),  
                                net_idt.d_64.sentence_attn_16.parameters(), 
                                ), 'lr': 0.2*args.lr})

        if i==( (options.total_iter//4) * 3):
            opt_idt.add_param_group({'params': chain(net_idt.d_128.word_attn_4.parameters(), 
                                net_idt.d_128.word_attn_16.parameters(),
                                net_idt.d_128.sentence_attn_4.parameters(),  
                                net_idt.d_128.sentence_attn_16.parameters()
                                ), 'lr': 0.2*args.lr})
        ### 2.1 train the image-only discriminator
        
        net_id.zero_grad()

        real_predict = net_id(real_image, step)
        fake_predict_from_real_text = net_id(g_image_from_real_text.detach(), step)
        fake_predict_from_g_text = net_id(g_image_from_g_text.detach(), step)

        loss_disc = F.relu(1-real_predict).mean() + \
                        F.relu(1+fake_predict_from_real_text).mean() +\
                            F.relu(1+fake_predict_from_g_text).mean()  
        loss_disc.backward()
        opt_id.step()

        image_d_val += real_predict.mean().item()

        ### 2.2 train the image-text discriminator
        net_idt.zero_grad()

        real_predict_s, real_predict_w = net_idt(real_image, real_text_latent, real_embs, step)
        fake_predict_sr, fake_predict_wr = net_idt(g_image_from_real_text.detach(), real_text_latent, real_embs, step)
        fake_predict_sg, fake_predict_wg = net_idt(g_image_from_g_text.detach(), g_text_latent.detach(), g_embs.detach(), step)
        
        mismatch_predict_s, mismatch_predict_w = net_idt(real_image, real_text_latent[perm], real_embs[perm], step)

        loss_disc = 2*F.relu(1-real_predict_s).mean() + 2*F.relu(1-real_predict_w).mean() + \
                    F.relu(1+fake_predict_sr).mean() + F.relu(1+fake_predict_wr).mean() + \
                    F.relu(1+fake_predict_sg).mean() + F.relu(1+fake_predict_wg).mean() + \
                    F.relu(1+mismatch_predict_s).mean() + F.relu(1+mismatch_predict_w).mean()

        loss_disc.backward()
        opt_idt.step()
        
        image_dt_val += real_predict_s.mean().item() + real_predict_w.mean().item()
        image_dt_mis_val += mismatch_predict_s.mean().item() + mismatch_predict_w.mean().item()
        
        ### 2.3 train the text discriminator
        net_td.zero_grad()

        real_predict_latent = net_td(real_text_latent_with_noise.detach())
        fake_predict_latent = net_td(g_text_latent.detach())
        
        loss_disc = F.relu(1-real_predict_latent).mean() + F.relu(1+fake_predict_latent).mean() 
        loss_disc.backward()
        opt_td.step()

        text_d_val += real_predict_latent.mean().item()

        # 3. Train the Generators

        if i==(options.total_iter//4):
            opt_ig.add_param_group({'params': chain(net_ig.word_attn_4.parameters(), 
                                net_ig.word_attn_8.parameters(),
                                net_ig.word_attn_16.parameters(), 
                                net_ig.sentence_attn_4.parameters(),  
                                net_ig.sentence_attn_8.parameters(),  
                                net_ig.sentence_attn_16.parameters(), 
                                ), 'lr': 0.2*args.lr})


        net_ig.zero_grad()
        net_tg.zero_grad()
        
        g_pred_rt = net_id(g_image_from_real_text, step)
        g_pred_sr, g_pred_wr = net_idt(g_image_from_real_text, real_text_latent.detach(), real_embs.detach(), step)

        loss_g_image = -g_pred_rt.mean() - g_pred_sr.mean() - g_pred_wr.mean()

        if i > 10000:
            g_pred_gt = net_id(g_image_from_g_text, step)
            g_pred_sg, g_pred_wg = net_idt(g_image_from_g_text, g_text_latent, g_embs, step)

            loss_g_image = loss_g_image - g_pred_gt.mean() - g_pred_sg.mean() - g_pred_wg.mean()

        loss_g_text = - net_td(g_text_latent).mean()
        
        #loss_g_text.backward()
        loss_total = loss_g_image + loss_g_text
        loss_total.backward()
        
        opt_ig.step()
        opt_tg.step()

        image_g_val += g_pred_rt.mean().item()
        image_gt_val += g_pred_sr.mean().item() + g_pred_wr.mean().item()
        text_g_val += -loss_g_text.item()
        
        ### 4. Logging 
        if (i + 1) % 2000 == 0 or i==0:
            with torch.no_grad():            
                vutils.save_image(g_image_from_real_text.detach().add(1).mul(0.5), f'{log_folder}/sample/gr_img_{str(i + 1).zfill(6)}.jpg')
                real_texts = net_t_ae.generate(real_text_latent)
                f = open(f'{log_folder}/sample/gr_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for cap in real_texts:
                    f.write(cap+'\n')
                f.close()    

                vutils.save_image(g_image_from_g_text.detach().add(1).mul(0.5), f'{log_folder}/sample/gg_img_{str(i + 1).zfill(6)}.jpg')
                g_texts = net_t_ae.generate(g_text_latent)
                f = open(f'{log_folder}/sample/gg_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for cap in g_texts:
                    f.write(cap+'\n')
                f.close()  


        if (i+1) % 5000 == 0 or i==0:
            torch.save({'ig':net_ig.state_dict(), 'id':net_id.state_dict(), 'idt':net_idt.state_dict(),
                        'tg':net_tg.state_dict(), 'td':net_td.state_dict()}, f'{log_folder}/checkpoint/image_to_text_memory_{str(i + 1).zfill(6)}_model.pth')
            torch.save({'ig':opt_ig.state_dict(), 'id':opt_id.state_dict(), 'idt':opt_idt.state_dict(),
                        'tg':opt_tg.state_dict(), 'td':opt_td.state_dict()}, f'{log_folder}/checkpoint/image_to_text_memory_{str(i + 1).zfill(6)}_opt.pth')
            
        interval = 100
        if (i+1)%interval == 0:
            
            state_msg = (f'img_g_val: {image_g_val/(interval):.3f};   img_d_val: {image_d_val/interval:.3f};   \n'
                        f'img_gt_val: {image_gt_val/(interval):.3f};   img_dt_val: {image_dt_val/interval:.3f};  img_dt_mis: {image_dt_mis_val/interval:.3f} \n'
                        f' Text_g: {text_g_val/(200):.3f};   Text_d: {text_d_val/200:.3f};\n')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f,%.5f\n"%\
                (image_g_val/(interval), image_d_val/interval, image_dt_val/interval)
            log_file.write(new_line)
            log_file.close()

            image_g_val = 0
            image_d_val = 0
            image_gt_val = 0
            image_dt_val = 0
            image_dt_mis_val = 0
            text_g_val = 0
            text_d_val = 0
            print(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Text Generation Together')

    parser.add_argument('--path', type=str, default='../../../research3/CUB_birds/CUB_200_2011/images', help='path of specified dataset')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--trial_name', default='trial_it_gan_with_pre-trained_sw', type=str, help='name of the trial')
    parser.add_argument('--total_iter', default=300000, type=int, help='iterations')
    parser.add_argument('--start_iter', default=0, type=int, help='start iterations')
    parser.add_argument('--im_size', default=128, type=int, help='initial image size')
    parser.add_argument('--batch_size', default=16, type=int, help='initial image size')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load pre-trained model')
    parser.add_argument('--channel', type=int, default=128, help='channel number in models')
    parser.add_argument('--ae_path', type=str, default=None, help='path to load pre-trained text Autoencoder model')
    
    args = parser.parse_args()

    img_meta_root = str(args.path).replace('images','birds_meta')
    
    # 0. Create the pre-trained Text Autoencoder model
    pre_trained_path = './it_ae_160000_model.pth'
    checkpoint = torch.load(pre_trained_path)
    net_t_ae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=img_meta_root).cuda()
    net_t_ae.load_state_dict(checkpoint['t'])
    net_t_ae.eval()
    for p in net_t_ae.parameters():
        p.requires_grad = False


    # 1. Create the Text GAN 
    net_tg = Text_Latent_G(latent=128, noise=128).cuda()
    net_td = Text_Latent_D(latent=128).cuda()


    # 2. Create the Image GAN
    net_ig = G()
    net_ig.cuda()

    net_ig.sentence_attn_4.load_state_dict(checkpoint['sa4'])
    net_ig.sentence_attn_8.load_state_dict(checkpoint['sa8'])
    net_ig.sentence_attn_16.load_state_dict(checkpoint['sa16'])
    net_ig.word_attn_4.load_state_dict(checkpoint['wa4'])
    net_ig.word_attn_8.load_state_dict(checkpoint['wa8'])
    net_ig.word_attn_16.load_state_dict(checkpoint['wa16'])


    net_id = D()
    net_id.cuda()

    # 3. Create the Image-Text matching GAN
    net_idt = D_with_t()
    net_idt.cuda()

    net_idt.d_64.word_attn_4.load_state_dict(checkpoint['wa4'])
    net_idt.d_64.sentence_attn_4.load_state_dict(checkpoint['sa4'])
    net_idt.d_64.word_attn_16.load_state_dict(checkpoint['wa16'])
    net_idt.d_64.sentence_attn_16.load_state_dict(checkpoint['sa16'])

    net_idt.d_128.word_attn_4.load_state_dict(checkpoint['wa4'])
    net_idt.d_128.sentence_attn_4.load_state_dict(checkpoint['sa4'])
    net_idt.d_128.word_attn_16.load_state_dict(checkpoint['wa16'])
    net_idt.d_128.sentence_attn_16.load_state_dict(checkpoint['sa16'])


    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net_tg.load_state_dict(checkpoint['tg'])
        net_td.load_state_dict(checkpoint['td'])

        net_ig.load_state_dict(checkpoint['ig'])
        net_id.load_state_dict(checkpoint['id'])
        net_idt.load_state_dict(checkpoint['idt'])

    # 4. create the optimizers
    opt_tg = optim.Adam( net_tg.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_td = optim.Adam( net_td.parameters(), lr=args.lr, betas=(0.5, 0.999))

    opt_ig = optim.Adam( chain( net_ig.conv_4.parameters(),
                                net_ig.conv_8.parameters(),
                                net_ig.conv_16.parameters(),
                                net_ig.conv_32.parameters(),
                                net_ig.conv_64.parameters(),
                                net_ig.conv_128.parameters(),
                                net_ig.to_rgb_64.parameters(),
                                net_ig.to_rgb_128.parameters(),
                                net_ig.sentence_rec_4.parameters(),
                                net_ig.sentence_rec_8.parameters(),
                                net_ig.sentence_rec_16.parameters(),
                                net_ig.word_rec_4.parameters(),
                                net_ig.word_rec_8.parameters(),
                                net_ig.word_rec_16.parameters(),
                                ), lr=args.lr, betas=(0.5, 0.99))
    '''
    opt_ig.add_param_group({'params': chain(net_ig.word_attn_4.parameters(), 
                                net_ig.word_attn_8.parameters(),
                                net_ig.word_attn_16.parameters(), 
                                net_ig.sentence_attn_4.parameters(),  
                                net_ig.sentence_attn_8.parameters(),  
                                net_ig.sentence_attn_16.parameters(), 
                                ), 'lr': 0.1*args.lr})
    '''
    opt_id = optim.Adam( net_id.parameters(), lr=args.lr, betas=(0.5, 0.99))
    
    opt_idt = optim.Adam( chain(net_idt.d_64.conv_4.parameters(),net_idt.d_64.conv_8.parameters(),
                                net_idt.d_64.conv_16.parameters(),net_idt.d_64.conv_32.parameters(),
                                net_idt.d_64.rf_word.parameters(),net_idt.d_64.rf_sentence.parameters(),
                                net_idt.d_128.conv_4.parameters(),net_idt.d_128.conv_8.parameters(),
                                net_idt.d_128.conv_16.parameters(),net_idt.d_128.conv_32.parameters(),
                                net_idt.d_128.conv_64.parameters(),
                                net_idt.d_128.rf_word.parameters(),net_idt.d_128.rf_sentence.parameters(),
                                ), lr=args.lr, betas=(0.5, 0.99))
    
    '''
    opt_idt.add_param_group({'params': chain(net_idt.d_64.word_attn_4.parameters(), 
                                net_idt.d_64.word_attn_16.parameters(),
                                net_idt.d_64.sentence_attn_4.parameters(),  
                                net_idt.d_64.sentence_attn_16.parameters(), 
                                net_idt.d_128.word_attn_4.parameters(), 
                                net_idt.d_128.word_attn_16.parameters(),
                                net_idt.d_128.sentence_attn_4.parameters(),  
                                net_idt.d_128.sentence_attn_16.parameters(), 
                                ), 'lr': 0.1*args.lr})
    '''

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint.replace('model.pth', 'opt.pth'))
        opt_tg.load_state_dict(checkpoint['tg'])
        opt_td.load_state_dict(checkpoint['td'])
        opt_ig.load_state_dict(checkpoint['ig'])
        opt_id.load_state_dict(checkpoint['id'])
        opt_idt.load_state_dict(checkpoint['idt'])

    print(args.path)
    loader = image_cap_loader(args.path)

    total_iter = args.total_iter
    train_image_text_gan(net_ig, net_id, net_tg, net_td, net_idt, total_iter, loader, args)