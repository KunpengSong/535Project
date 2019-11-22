import torch.utils.data as data
import os, pickle
from PIL import Image
import torch
from torchvision import datasets, transforms

#img_meta_root = '/media/bingchen/research3/CUB_birds/CUB_200_2011/birds_meta'
#img_root = '/media/bingchen/research3/CUB_birds/CUB_200_2011/images'
transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class CaptionDataset(data.Dataset):
    def __init__(self, img_meta_root, transform=None):
        super().__init__()

        self.filenames_caption_ids = pickle.load(open(img_meta_root+'/file_with_bert_dic.pkl', 'rb'))
        #self.filename_bert_embs = pickle.load((open(img_meta_root+'/file_with_bert_emb.pkl', 'rb'))) 
        #self.filenames_caption_ids = self.filenames_caption_ids[:10]
        #self.embs = pickle.load(open(img_meta_root+'/file_with_bert_emb.pkl', 'rb'))
    def __len__(self):
        return len(self.filenames_caption_ids)

    def __getitem__(self, idx): 
        bird = torch.tensor(self.filenames_caption_ids[idx].get('bird'))
        bert = torch.tensor(self.filenames_caption_ids[idx].get('bert'))
        return bird, bert


class CaptionImageDataset(data.Dataset):
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
        bert = torch.tensor(self.filenames_caption_ids[idx].get('bert'))
        return im, bird, bert


class EmbDataset(data.Dataset):
    def __init__(self, img_root, img_meta_root, transform=None):
        super().__init__()

        self.transform = transform

        self.img_root = img_root
        print('Loading image caption embeddings ...')
        self.filenames_with_emb = pickle.load(open(img_meta_root, 'rb'))
        print('Done.')

    def __len__(self):
        return len(self.filenames_with_emb)

    def __getitem__(self, idx): 
        fc = self.filenames_with_emb[idx]
        im = Image.open(os.path.join(self.img_root, fc[0]+'.jpg')).convert('RGB')
        
        emb = fc[1].squeeze(0)
        if self.transform is not None:
            im = self.transform(im)
        return (im, emb)

'''
img_root = '/media/bingchen/research3/CUB_birds/CUB_200_2011/images'
img_meta_root = '/media/bingchen/research3/CUB_birds/CUB_200_2011/birds_meta/file_with_bert_emb.pkl'
dataset = EmbDataset(img_root, img_meta_root)
dataset[1]
'''
#dataset = CaptionDataset()