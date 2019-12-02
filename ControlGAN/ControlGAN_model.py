import torch
from torch import nn
from modules import UnFlatten, GLU, upBlock, Spatial_Attn, Channel_Attn, Decoupling_Attn,Upscale2d
from modules import ResBlock, conv1x1, conv3x3, block3x3, weights_init
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from utils import true_randperm
from modules import downBlock
import torch.nn.functional as F
from utils import true_randperm
from pytorch_transformers import BertModel, BertTokenizer, BertConfig


class Text_bert(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig()
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_word_emb(self,words): #tensor: b,15
        embs = self.bert(words)
        word_embs = embs[0] #b,15,768
        mem = embs[2] #tuple: length=13, each: b,15,768, last layer same as word_embs
        return word_embs,mem
    
    def get_sentence_emb(self,mem): # use [4,8,10,12] from [0,1,...,12]
        mem4,mem8,mem10,mem12 = mem[4],mem[8],mem[10],mem[12]
        
        sentence = torch.cat([mem4,mem8,mem10,mem12],dim=-2).permute(0,2,1) #b,15,768 --> b,60,768 --> b,768,60
        #sentence = nn.MaxPool1d(60)(sentence).squeeze(-1) #b,768,60 --> b,768,1 --> b,768
        sentence = torch.nn.functional.adaptive_avg_pool2d(sentence,(256,1)).squeeze(-1) #b,768,60 --> b,256,1 --> b,256
        return sentence


class Attn(nn.Module):
    def __init__(self, word_dim=768, img_hw=32, img_channel=128):
        super().__init__()

        self.channel_attn = Channel_Attn(word_dim, img_channel, img_hw)
        self.spatial_attn = Spatial_Attn(word_dim, img_channel, img_hw)

        self.conv = block3x3(img_channel*2, img_channel)

    def forward(self, img_feat, word_emb):
        img_feat_c = self.channel_attn(img_feat,word_emb)
        img_feat_s = self.spatial_attn(img_feat,word_emb)
        feat = torch.cat([img_feat_c, img_feat_s], dim=1)
        return self.conv(feat)


class G(nn.Module):
    def __init__(self, channel=512, noise_dim=256, sentence_dim=256, word_dim=768):
        super().__init__()

        self.conv_8 = nn.Sequential(
                        UnFlatten(),
                        nn.ConvTranspose2d(noise_dim+sentence_dim, channel*2, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(channel*2), 
                        GLU(),
                        upBlock(channel, channel//2))

        self.attn_8 = Attn(word_dim, 8, channel//2)

        self.conv_32 = nn.Sequential(
                        upBlock(channel//2, channel//2),
                        upBlock(channel//2, channel//4))

        self.attn_32 = Attn(word_dim, 32, channel//4)
        
        self.conv_64 = upBlock(channel//4, channel//4)
        self.attn_64 = Attn(word_dim, 64, channel//4)        

        self.conv_256 = nn.Sequential(
                        upBlock(channel//4, channel//8),                        
                        upBlock(channel//8, channel//16))

        self.to_rgb_64 = nn.Sequential(conv3x3(channel//4, 3), nn.Tanh())
        self.to_rgb_256 = nn.Sequential(conv3x3(channel//16, 3), nn.Tanh())

        self.apply(weights_init)

    def forward(self, noise,sentence,word_emb): #b,256; b,256; 4, 15, 768
        feat = torch.cat([noise,sentence],dim=-1).unsqueeze(-1).unsqueeze(-1) #b,256; b,256 --> b,512,1,1
        feat_8 = self.conv_8(feat)
        feat_8 = self.attn_8(feat_8,word_emb)
        feat_32 = self.conv_32(feat_8) 
        feat_32 = self.attn_32(feat_32, word_emb)
        feat_64 = self.conv_64(feat_32)
        feat_64 = self.attn_64(feat_64, word_emb)
        feat_256 = self.conv_256(feat_64)
        
        return [self.to_rgb_64(feat_64), self.to_rgb_256(feat_256)]


class D(nn.Module):
    def __init__(self, channel=512, sentence_dim = 256, word_dim=768):
        super().__init__()

        self.from_256 = nn.Sequential(
            downBlock(3, channel//8),
            downBlock(channel//8, channel//4),
            downBlock(channel//4, channel//4),
            downBlock(channel//4, channel//2),
            downBlock(channel//2, channel//2),
        )

        self.from_64 = nn.Sequential( 
            downBlock(3, channel//8),
            downBlock(channel//8, channel//4),
            downBlock(channel//4, channel//2),
        )

        self.rf_64 = nn.Sequential(
            downBlock(channel//2, channel),
            spectral_norm(nn.Conv2d(channel, 1, 4, 1, 0, bias=True))
        )

        self.rf_256 = nn.Sequential(
            downBlock(channel//2, channel),
            spectral_norm(nn.Conv2d(channel, 1, 4, 1, 0, bias=True))
        )

        #self.it_match = TextImageMatcher(channel//2, word_dim)

        self.matcher = nn.Sequential(
            downBlock(channel+sentence_dim, channel),
            spectral_norm(nn.Conv2d(channel, channel//4, 3, 1, 1, bias=True)),
            spectral_norm(nn.Conv2d(channel//4, 1, 4, 1, 0, bias=True)),
        )

        self.apply(weights_init)


    def forward(self, imgs, sentence=None, word_emb=None, train_perm=True):
        feat_64 = self.from_64(imgs[0]) 
        feat_256 = self.from_256(imgs[1])

        logit_1 = self.rf_64(feat_64).view(-1)
        logit_3 = self.rf_256(feat_256).view(-1)
        
        b = imgs[0].shape[0]
        img_feat = torch.cat([feat_64, feat_256], dim=1)

        match = pred_text = match_perm = pred_text_perm = None

        #match = self.it_match(img_feat, sentence)
        match = self.matcher(torch.cat([img_feat, sentence.unsqueeze(-1).unsqueeze(-1).repeat(1,1,8,8)],dim=1)) #b,256

        if train_perm:
            perm = true_randperm(b)
            #match_perm = self.it_match(img_feat[perm], sentence)
            match_perm = self.matcher(torch.cat([img_feat[perm], sentence.unsqueeze(-1).unsqueeze(-1).repeat(1,1,8,8)],dim=1)) #b,256

        return torch.cat([logit_1, logit_3]), match, match_perm


class TextImageMatcher(nn.Module):
    def __init__(self, img_channel, sentence):
        super().__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channel*2, img_channel, 3, 1, 1, bias=True, groups=2)),
            nn.BatchNorm2d(img_channel),
            nn.AdaptiveAvgPool2d(4),
            nn.LeakyReLU(0.2),
        )

        self.rf = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channel + sentence, img_channel, 3, 1, 1, bias=True)),
            nn.BatchNorm2d(img_channel),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(img_channel, 1, 4, 1, 0, bias=False))
        )

    def forward(self, img_feat, text_emb): #sentence:b,256
        text_emb = text_emb.permute(1,2,0)
        b, c, t_len = text_emb.shape
        if t_len < 16:
            text_emb = torch.cat([text_emb, torch.zeros(b, c, 16-t_len).to(text_emb.device)], dim=-1)
        elif t_len > 16:
            text_emb = text_emb[:,:,1:17]
        text_emb = text_emb.view(b, c, 4, 4)
        return self.rf( torch.cat((self.conv(img_feat), text_emb), dim=1) ).view(-1)

