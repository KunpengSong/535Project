import torch.utils.data as data
import os, pickle
from PIL import Image
import torch
from torchvision import datasets, transforms
from random import shuffle
import torch.nn.functional as F
from torch import nn
from torch import autograd
from copy import deepcopy


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

LAMBDA = 10 # Gradient penalty lambda hyperparameter.
def calc_gradient_penalty(netD, real_data, fake_data, text_emb, text_mask, train_perm=False, train_trec=False, size=64):
    interpolates = []
    for i in range(len(real_data)):
        if fake_data[i] is None:
            interpolates.append(torch.zeros_like(real_data[i], requires_grad=True))
        else:
            alpha = torch.rand(real_data[i].shape[0], 1, 1, 1).to(real_data[0].device)
            interpolate = alpha * real_data[i].data + ((1 - alpha) * fake_data[i].data)
            interpolate.requires_grad = True
            interpolates.append(interpolate)
    hat_predict, _,_,_,_ = netD(interpolates, text_emb, text_mask, train_perm, train_trec, size)
    gradients = autograd.grad(outputs=hat_predict.sum(), inputs=interpolates,
                              create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

    gradient_penalty = ((gradients.view(real_data[0].shape[0], -1)
                        .norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def load_eval_model(net : nn.Module, state_dict, device):
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    for p in net.parameters():
        p.requires_grad = False

def trans_maker(image_size):
    transform = transforms.Compose([
        #transforms.Resize(image_size))
        transforms.Resize( int(1.1 * image_size) ),
        #transforms.CenterCrop( int(1.2 * image_size) ),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_path(log_folder):
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        os.mkdir(log_folder+'/checkpoint')
        os.mkdir(log_folder+'/sample')

    from shutil import copy
    try:
        for f in os.listdir('.'):
            if '.py' in f:
                copy(f, log_folder+'/'+f)
    except:
        pass

    return log_folder
    
def resize(feat, size=64):
    return F.interpolate(feat, size=size, mode='nearest')

def detach(feat):
    if type(feat) is list or type(feat) is tuple:
        result = []
        for f in feat:
            if f is not None:
                f=f.detach()
            result.append(f)
        return result
    else:
        return feat.detach()

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

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


class CaptionImageDataset(data.Dataset):
    def __init__(self, img_root, img_meta_path, transform=None):
        super().__init__()

        self.transform = transform
        self.img_root = img_root

        self.frame = pickle.load(open(img_meta_path, 'rb'))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx): 
        img_name = self.frame[idx][0]+'.jpg'
        im = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        cap_idx = torch.tensor(self.frame[idx][1])
        return im, cap_idx


class BirdCaptionImageDataset(data.Dataset):
    def __init__(self, img_root, img_meta_root, transform=None):
        super().__init__()

        self.transform = transform
        self.img_root = img_root

        self.filenames_caption_ids = pickle.load(open(img_meta_root+'/file_with_bert_dic.pkl', 'rb'))

    def __len__(self):
        return len(self.filenames_caption_ids)

    def __getitem__(self, idx): 
        img_name = self.filenames_caption_ids[idx].get('file')
        im = Image.open(os.path.join(self.img_root, img_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        bird = torch.tensor(self.filenames_caption_ids[idx].get('bird'))
        #bert = torch.tensor(self.filenames_caption_ids[idx].get('bert'))
        return im, bird#, bert

from torch.nn.utils.rnn import pad_sequence
def pad_packed_collate(batch):

    tuple_of_seq = [pair[1] for pair in batch]
    padded_seq = pad_sequence(tuple_of_seq, padding_value=1)
    
    images = torch.stack( [pair[0] for pair in batch], dim=0)
    return images, padded_seq.permute(1, 0)


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count