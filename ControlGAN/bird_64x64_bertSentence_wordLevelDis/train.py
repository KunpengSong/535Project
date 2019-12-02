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

from pytorch_transformers import BertModel, BertTokenizer
from ControlGAN_model import Text_bert,G,D



torch.backends.cudnn.benchmark = True

image_size = 64
batch_size = 24
device = torch.device('cuda')

log_interval = 100
nepoch = 60
grad_clip = 0.25
trail_name = 'bird_64x64_OnlyWord'
log_folder = get_path(trail_name)
start_epoch = 0


pretrained_bert = Text_bert().to(device)


img_root='../../data/CUB_200_2011/images'
img_meta_root='../../data/file_with_bert_cap.pkl'

data = CaptionImageDataset(img_root, img_meta_root, transform=trans_maker(image_size))
data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)

net_ig = G().to(device)
net_id = D().to(device)

net_ig.to(device)
net_ig = nn.DataParallel(net_ig)
opt_ig = optim.Adam(net_ig.parameters(), lr=1e-4, betas=(0.5,0.999))

net_id.to(device)
net_id = nn.DataParallel(net_id)
opt_id = optim.Adam(net_id.parameters(), lr=1e-4, betas=(0.5,0.999))

avg_param_G = copy_G_params(net_ig)


losses_gp = AverageMeter()
losses_g_img = AverageMeter()
losses_g_match = AverageMeter()
losses_d_img = AverageMeter()
losses_d_match = AverageMeter()
losses_d_mismatch = AverageMeter()

losses_d_word_match = AverageMeter()
losses_d_word_mismatch = AverageMeter()
losses_g_word_match = AverageMeter()

fixed_inp = None
fixed_wordidx = None


for epoch in range(start_epoch, nepoch):

    for batch_idx, (real_image, word_idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
        real_image = real_image.to(device)
        real_image = [resize(real_image, 64), real_image]
        
        word_idx = word_idx.to(device)
        batch_size = word_idx.shape[0]

        word_emb,mem = pretrained_bert.get_word_emb(word_idx) #b, 15, 768
        sentence = pretrained_bert.get_sentence_emb(mem) #b,256
        word_emb,sentence = word_emb.to(device),sentence.to(device)
        #print(word_emb,sentence)
        #assert 0 == 1
        
        noise = torch.randn(batch_size, 256).to(device)

        if fixed_inp is None:
            fixed_inp = noise.clone()
            fixed_wordidx = word_idx.clone()
            vutils.save_image( real_image[-1].clone().add(1).mul(0.5), f'{log_folder}/sample/real_img.jpg')
        
        g_image = net_ig(noise,sentence.detach(),word_emb.detach())

        ### 1. train D
        net_id.zero_grad()

        pred_r, pred_match, pred_mismatch, word_match, word_mismatch = net_id(real_image, sentence=sentence.detach(), word_emb=word_emb.detach(), train_perm=True)

        '''
        loss_r = 2*F.relu(1 - pred_r).mean() + 0.001 * (pred_r ** 2).mean() +\
                    F.relu(1 - pred_match).mean() + 0.001 * (pred_match ** 2).mean() +\
                        F.relu(1 - word_match).mean() + 0.001 * (word_match ** 2).mean() +\
                            F.relu(1 + pred_mismatch).mean() +\
                                F.relu(1 + word_mismatch).mean()
        '''
        #loss_r = 2*F.relu(1 - pred_r).mean() + 0.001 * (pred_r ** 2).mean()
        #loss_r = 2*F.relu(1 - pred_r).mean() + 0.001 * (pred_r ** 2).mean() +\
        #            F.relu(1 - pred_match).mean() + 0.001 * (pred_match ** 2).mean() +\
        #                    F.relu(1 + pred_mismatch).mean()
        loss_r = 2*F.relu(1 - pred_r).mean() + 0.001 * (pred_r ** 2).mean() +\
                        F.relu(1 - word_match).mean() + 0.001 * (word_match ** 2).mean() +\
                                F.relu(1 + word_mismatch).mean()

        weight_fmatch = 1
        if epoch > 10:
            weight_fmatch = 0.2
        pred_f, pred_fmatch, _, word_fmatch, _  = net_id(detach(g_image), sentence=sentence.detach(), word_emb=word_emb.detach(), train_perm=False)
        
        '''
        loss_f = F.relu(1 + pred_f).mean() + weight_fmatch * F.relu(1 + pred_fmatch).mean()
        '''
        #loss_f = F.relu(1 + pred_f).mean()
        #loss_f = 2*F.relu(1 + pred_f).mean() + weight_fmatch * F.relu(1 + pred_fmatch).mean()
        loss_f = 2*F.relu(1 + pred_f).mean() + weight_fmatch * F.relu(1 + word_fmatch).mean()


        
        loss_d = loss_r + loss_f
        loss_d.backward()
        

        LAMBDA = 10 # Gradient penalty lambda hyperparameter.
        gradient_penalty = None
        '''
        if batch_size >= 8:
            alpha = torch.rand(8, 1, 1, 1).to(device)
            interpolates = []
            for img_idx in range(len(g_image)):
                interpolate = alpha * real_image[img_idx][:8].data + ((1 - alpha) * g_image[img_idx][:8].data)
                interpolate.requires_grad = True
                interpolates.append(interpolate)
            hat_predicts = net_id(interpolates, sentence=sentence[:8,:].detach(),word_emb=word_emb[:8,:,:].detach(),train_perm=False)
            image_pred = hat_predicts[0]
            itmatch_pred = hat_predicts[1]
            image_gradients = autograd.grad(outputs=image_pred.sum(), inputs=interpolates,
                                create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
            itmatch_gradients = autograd.grad(outputs=itmatch_pred.sum(), inputs=interpolates,
                                create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
            gradient_penalty = ((image_gradients.view(8, -1).norm(2, dim=1) - 1) ** 2).mean() * LAMBDA +\
                                    ((itmatch_gradients.view(8, -1).norm(2, dim=1) - 1) ** 2).mean() * LAMBDA * 0.5
            gradient_penalty.backward()
        '''

        opt_id.step()


        ### 2. train G
        net_ig.zero_grad()

        pred_g, pred_gmatch, _, word_gmatch, _= net_id(g_image, sentence=sentence.detach(), word_emb=word_emb.detach(), train_perm=False)

        '''
        loss_g = -pred_g.mean() - \
                    pred_gmatch.mean() - \
                        word_gmatch.mean()
        '''
        #loss_g = -pred_g.mean()
        loss_g = -2*pred_g.mean() - \
                    word_gmatch.mean()



        loss_g.backward()

        torch.nn.utils.clip_grad_norm_(net_ig.parameters(), grad_clip)
        opt_ig.step()

        for p, avg_p in zip(net_ig.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001, p.data)

        
        if gradient_penalty:
            losses_gp.update(gradient_penalty.item(), batch_size)

        losses_g_img.update(pred_g.mean().item(), batch_size)
        #losses_g_match.update(pred_gmatch.mean().item(), batch_size)
        losses_d_img.update(pred_r.mean().item(), batch_size)
        #losses_d_match.update(pred_match.mean().item(), batch_size)
        #losses_d_mismatch.update(pred_mismatch.mean().item(), batch_size)

        
        losses_d_word_mismatch.update(word_mismatch.mean().item(), batch_size)
        losses_d_word_match.update(word_match.mean().item(), batch_size)
        losses_g_word_match.update(word_gmatch.mean().item(), batch_size)
        
        
        if batch_idx % log_interval == 0:
            print('Epoch: [{0}/{1}]\t'
                    '\nG: img: {losses_g_img.avg:.4f}  match: {losses_g_match.avg:.4f}  gradient-penalty: {losses_gp.avg:.4f}'
                    '\nD: img: {losses_d_img.avg:.4f}  match: {losses_d_match.avg:.4f}  mismatch: {losses_d_mismatch.avg:.4f}'
                    '\nD: w_mismatch: {losses_d_word_mismatch.avg:.4f}  w_match: {losses_d_word_match.avg:.4f} G: w_match:{losses_g_word_match.avg:.4f}'

                    .format(epoch, nepoch,
                    losses_g_img=losses_g_img, losses_g_match=losses_g_match, losses_gp=losses_gp,
                    losses_d_img=losses_d_img, losses_d_match=losses_d_match, losses_d_mismatch=losses_d_mismatch,
                    losses_d_word_mismatch = losses_d_word_mismatch, losses_d_word_match = losses_d_word_match,losses_g_word_match = losses_g_word_match,
                    ))
            
            losses_gp.reset()
            losses_g_img.reset()
            losses_g_match.reset()

            losses_d_img.reset()
            losses_d_match.reset()
            losses_d_mismatch.reset()

            losses_d_word_mismatch.reset()
            losses_d_word_match.reset()
            losses_g_word_match.reset()

        
        if batch_idx % (log_interval) == 0:
            with torch.no_grad():
                fixed_word_emb,fixed_mem = pretrained_bert.get_word_emb(fixed_wordidx) #b, 15, 768
                fixed_sentence = pretrained_bert.get_sentence_emb(fixed_mem) #b,256
                fixed_word_emb,fixed_sentence = fixed_word_emb.to(device),fixed_sentence.to(device)

                backup_para = copy_G_params(net_ig)
                load_params(net_ig, avg_param_G)

                fixed_img = net_ig(fixed_inp,fixed_sentence.detach(),fixed_word_emb.detach())
                vutils.save_image(fixed_img[-1].add(1).mul(0.5), f'{log_folder}/sample/img_epoch_{str(epoch)}_iter_{str(batch_idx).zfill(6)}.png')
                vutils.save_image(fixed_img[-1].add(1).mul(0.5), f'{log_folder}/sample/{trail_name}_latest.png')

                load_params(net_ig, backup_para)


    if epoch%1==0:
        backup_para = copy_G_params(net_ig)
        load_params(net_ig, avg_param_G)
        torch.save( {'ig': net_ig.state_dict(), 
                     'id': net_id.state_dict()}, '%s/checkpoint/%d.pth'%(log_folder ,epoch))
        torch.save( {'ig': opt_ig.state_dict(), 
                    'id': opt_id.state_dict()}, '%s/checkpoint/%d_opt.pth'%(log_folder ,epoch))
        load_params(net_ig, backup_para)

