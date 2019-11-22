from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import pickle
import os
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
from models import ImageAE, SentenceAttention, WordAttention
from text_models import Text_VAE, Text_fuse_G, Text_fuse_D

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
        #transforms.CenterCrop( int(1.1 * image_size) ),
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


def train_image_text_ae(net_iae, net_tae, opt_iae, opt_tae, total_iter, loader, options):
    

    image_rec_val = 0
    text_emb_rec_val = 0
    text_idx_rec_val = 0
    sa_match_val = 0
    wa_match_val = 0
    sa_mismatch_val = 0
    wa_mismatch_val = 0

    log_folder = options.trial_name
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')
    
    log_file_name = os.path.join(log_folder, 'train_image_to_text_log.txt')
    log_file = open(log_file_name, 'w')
    log_file.write('rec, prob, code\n')
    log_file.close()

    copy('train_iae.py', log_folder+'/train_iae.py')
    copy('models.py', log_folder+'/models.py')


    data_loader = sample_data(loader, image_size=options.im_size, batch_size=options.batch_size)
    dataset = iter(data_loader)

    for i in tqdm(range(options.total_iter)):

        try:
            real_image, bird_idx, bert_idx = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, bird_idx, bert_idx = next(dataset)


        ### 1. load the data
        b_size = real_image.shape[0]
        real_image = real_image.cuda()
        bird_idx = bird_idx.long().cuda()
        bert_idx = bert_idx.cuda()
        perm = true_randperm(b_size)

        ### 2. init all the models
        net_tae.zero_grad()
        net_iae.zero_grad()
        sa_4.zero_grad()
        sa_8.zero_grad()
        sa_16.zero_grad()
        wa_4.zero_grad()
        wa_8.zero_grad()
        wa_16.zero_grad()

        '''
        ### 3. Forward pass of the text AE
        bert_embs = net_tae.bert(bert_idx.cuda())[0]
        text_latent = net_tae.encode(bert_embs)
        recon_text_logits, recon_embs = net_tae.decode(text_latent)

        '''
        ### 4. Forward pass of the image AE
        recon_img, img_feat_4, img_feat_8, img_feat_16 = net_iae(real_image)

        '''
        ### 5. Forward pass for the attention modules
        sa_4_match  = sa_4(img_feat_4, text_latent)
        sa_8_match  = sa_8(img_feat_8, text_latent)
        sa_16_match = sa_16(img_feat_16, text_latent)

        sa_4_mismatch  = sa_4(img_feat_4, text_latent[perm])
        sa_8_mismatch  = sa_8(img_feat_8, text_latent[perm])
        sa_16_mismatch = sa_16(img_feat_16, text_latent[perm])

        wa_4_match  = wa_4(img_feat_4, bert_embs)
        wa_8_match  = wa_8(img_feat_8, bert_embs)
        wa_16_match = wa_16(img_feat_16, bert_embs)

        wa_4_mismatch  = wa_4(img_feat_4, bert_embs[perm])
        wa_8_mismatch  = wa_8(img_feat_8, bert_embs[perm])
        wa_16_mismatch = wa_16(img_feat_16, bert_embs[perm])
        '''

        ### 6. Compute the losses
        '''''''''
        ## 6.1 loss for text ae
        loss_txt_idx_recon = F.cross_entropy(recon_text_logits, bird_idx.view(-1))
        loss_txt_emb_recon = F.l1_loss(recon_embs, bert_embs)
        loss_txt_latent_regularizer = F.l1_loss(text_latent.mean(), torch.tensor(0.0).cuda()) +\
                                        F.l1_loss(text_latent.std(), torch.tensor(1.0).cuda())

        loss_txt_total = loss_txt_idx_recon + loss_txt_emb_recon + loss_txt_latent_regularizer
        
        '''
        ## 6.2 loss for image ae
        loss_img_total = F.l1_loss(recon_img, real_image)
        
        '''
        ## 6.3 loss for attention modules
        loss_sentence = - sa_4_match.mean() + sa_4_mismatch.mean() - \
                            sa_8_match.mean() + sa_8_mismatch.mean() - \
                            sa_16_match.mean() + sa_16_mismatch.mean()
        loss_word = - wa_4_match.mean() + wa_4_mismatch.mean() - \
                            wa_8_match.mean() + wa_8_mismatch.mean() - \
                            wa_16_match.mean() + wa_16_mismatch.mean()
        loss_attn_total = loss_sentence + loss_word
        '''
        loss = loss_img_total #+ loss_txt_total + loss_attn_total
        loss.backward()

        opt_tae.step()
        opt_iae.step()
        opt_attn.step()

        ### 7. record the values
        image_rec_val += loss_img_total.item()
        #text_emb_rec_val += loss_txt_emb_recon.item()
        #text_idx_rec_val += loss_txt_idx_recon.item()
        #sa_match_val += sa_4_match.mean() + sa_8_match.mean() + sa_16_match.mean()
        #wa_match_val += sa_4_mismatch.mean() + sa_8_mismatch.mean() + sa_16_mismatch.mean()
        #sa_mismatch_val += wa_4_match.mean() + wa_8_match.mean() + wa_16_match.mean()
        #wa_mismatch_val += wa_4_mismatch.mean() + wa_8_mismatch.mean() + wa_16_mismatch.mean()

        ### 4. Logging 
        if (i + 1) % 2000 == 0 or i==0:
            with torch.no_grad():            
                
                vutils.save_image(recon_img.detach().add(1).mul(0.5), f'{log_folder}/sample/g_img_{str(i + 1).zfill(6)}.jpg')
                
                '''
                g_texts = net_tae.generate(text_latent)
                f = open(f'{log_folder}/sample/g_real_txt_{str(i + 1).zfill(6)}.txt', 'w')
                for cap in g_texts:
                    f.write(cap+'\n')
                f.close()    
                '''
        if (i+1) % 5000 == 0 or i==0:
            torch.save({'i':net_iae.state_dict(), \
                        't':net_tae.state_dict(), \
                        'sa4':sa_4.state_dict(), \
                        'sa8':sa_8.state_dict(), \
                        'sa16':sa_16.state_dict(), \
                        'wa4':wa_4.state_dict(), \
                        'wa8':wa_8.state_dict(), \
                        'wa16':wa_16.state_dict(), \
                        }, f'{log_folder}/checkpoint/it_ae_{str(i + 1).zfill(6)}_model.pth')
            
            torch.save({'i':opt_iae.state_dict(), \
                        't':opt_tae.state_dict(), \
                        'attn':opt_attn.state_dict() \
                        }, f'{log_folder}/checkpoint/it_ae_{str(i + 1).zfill(6)}_opt.pth')
            
        interval = 100
        if (i+1)%interval == 0:
            
            state_msg = (f'img_rec: {image_rec_val/(interval):.3f};   text_emb: {text_emb_rec_val/interval:.3f};   text_idx: {text_idx_rec_val/interval:.3f}\n'
                        f'sa: {sa_match_val/(interval):.3f};   sa_mis: {sa_mismatch_val/interval:.3f};  wa: {wa_match_val/(interval):.3f};   wa_mis: {wa_mismatch_val/interval:.3f} \n')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f,%.5f\n"%\
                (image_rec_val/(interval), text_idx_rec_val/interval, sa_match_val/interval)
            log_file.write(new_line)
            log_file.close()

            image_rec_val = 0
            text_emb_rec_val = 0
            text_idx_rec_val = 0
            sa_match_val = 0
            wa_match_val = 0
            sa_mismatch_val = 0
            wa_mismatch_val = 0
            print(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Text Generation Together')

    parser.add_argument('--path', type=str, default='../../../research3/CUB_birds/CUB_200_2011/images', help='path of specified dataset')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--trial_name', default='trial_i_ae', type=str, help='name of the trial')
    parser.add_argument('--total_iter', default=300000, type=int, help='iterations')
    parser.add_argument('--im_size', default=128, type=int, help='initial image size')
    parser.add_argument('--batch_size', default=8, type=int, help='initial image size')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load pre-trained model')
    parser.add_argument('--channel', type=int, default=128, help='channel number in models')
    parser.add_argument('--ae_path', type=str, default=None, help='path to load pre-trained text Autoencoder model')
    
    args = parser.parse_args()

    img_meta_root = str(args.path).replace('images','birds_meta')
    
    # creating Text model
    net_tae = Text_VAE(vocab_size=2098, channels=256, latent=128, meta_data_root=img_meta_root).cuda()
    opt_tae = optim.Adam( chain(net_tae.encoder.parameters(),\
                                    net_tae.decoder.parameters(),\
                                        net_tae.decoder_fc.parameters()) , lr=args.lr, betas=(0.5, 0.999))
    opt_tae.add_param_group({'params': net_tae.bert.parameters(), 'lr': args.lr * 0.1})

    # creating Image model
    net_iae = ImageAE(channel=256).cuda()
    opt_iae = optim.Adam(net_iae.parameters(), lr=0.1*args.lr, betas=(0.5, 0.999))
    
    # creating the attention models
    sa_4 = SentenceAttention(image_feat_channel=256, text_feat_channel=128, kernel_size=4).cuda()
    sa_8 = SentenceAttention(image_feat_channel=256, text_feat_channel=128, kernel_size=8).cuda()
    sa_16 = SentenceAttention(image_feat_channel=256, text_feat_channel=128, kernel_size=16).cuda()

    wa_4 = WordAttention(kernel_size=4).cuda()
    wa_8 = WordAttention(kernel_size=8).cuda()
    wa_16 = WordAttention(kernel_size=16).cuda()

    opt_attn = optim.Adam( chain(sa_4.parameters(), sa_8.parameters(), sa_16.parameters(),\
                                    wa_4.parameters(), wa_8.parameters(), wa_16.parameters()),\
                                        lr=args.lr, betas=(0.5, 0.999) )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net_tae.load_state_dict(checkpoint['t'])
        net_iae.load_state_dict(checkpoint['i'])
        sa_4.load_state_dict(checkpoint['sa4'])
        sa_8.load_state_dict(checkpoint['sa8'])
        sa_16.load_state_dict(checkpoint['sa16'])
        wa_4.load_state_dict(checkpoint['wa4'])
        wa_8.load_state_dict(checkpoint['wa8'])
        wa_16.load_state_dict(checkpoint['wa16'])

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint.replace('model.pth', 'opt.pth'))
        opt_tae.load_state_dict(checkpoint['t'])
        opt_iae.load_state_dict(checkpoint['i'])
        opt_attn.load_state_dict(checkpoint['attn'])

    print(args.path)
    loader = image_cap_loader(args.path)

    total_iter = args.total_iter
    train_image_text_ae(net_iae, net_tae, opt_iae, opt_tae, total_iter, loader, args)