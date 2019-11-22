import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UnFlatten(nn.Module):
    def __init__(self, block_size):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        feat = feat * torch.tanh(F.softplus(feat))
        return feat


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        
        self.conv = nn.Sequential(  spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)),
                                    nn.BatchNorm2d(channel), 
                                    nn.ReLU(),
                                    spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)),
                                    nn.BatchNorm2d(channel), 
                                    nn.ReLU() )
    def forward(self, feat):
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
        res = self.conv(feat)
        return feat + res



class SentenceAttention(nn.Module):
    def __init__(self, image_feat_channel=256, text_feat_channel=128, kernel_size=32):
        super().__init__()

        self.kernel_size = kernel_size

        self.key_from_img = nn.Conv2d(image_feat_channel, image_feat_channel//8, kernel_size=1)

        self.query_from_text = nn.Sequential(UnFlatten(1),
                        nn.ConvTranspose2d(text_feat_channel, image_feat_channel//8, 4, 1, 0),
                        nn.BatchNorm2d(image_feat_channel//8), nn.ReLU(),
                        nn.ConvTranspose2d(image_feat_channel//8, image_feat_channel//8, 4, 2, 1),
                        nn.AdaptiveAvgPool2d(kernel_size))
        
        self.value_from_img = nn.Conv2d(image_feat_channel, image_feat_channel, kernel_size=1)

        self.value_from_text = nn.Sequential(UnFlatten(1),
                        nn.ConvTranspose2d(text_feat_channel, image_feat_channel//4, 4, 1, 0),
                        nn.BatchNorm2d(image_feat_channel//4), nn.ReLU(),
                        nn.ConvTranspose2d(image_feat_channel//4, image_feat_channel, 4, 2, 1),
                        nn.AdaptiveAvgPool2d(kernel_size))

        self.softmax  = nn.Softmax(dim=-1)

    def read_feat(self, text_feat):
        b = text_feat.shape[0]
        
        proj_value_from_text = self.value_from_text(text_feat)

        return proj_value_from_text

    def read_img_feat(self, img_feat):
        b, c = img_feat.shape[:2]
        
        proj_key = self.key_from_img(img_feat)
        proj_key = F.adaptive_max_pool2d(proj_key, 1).view(b, c//8) # B X C  
        
        proj_v = self.value_from_img(img_feat)
        proj_v = F.adaptive_max_pool2d(proj_v, 1).view(b, c) # B X C  

        return torch.cat([proj_key, proj_v], dim=1)

    def rf(self, img_feat, text_feat):
        b, c, h, w = img_feat.shape

        proj_query  = self.query_from_text(text_feat).view(b,-1,h*w).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_from_img(img_feat).view(b,-1,h*w) # B X C x (N)   
        
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        
        proj_value_from_img = self.value_from_img(img_feat).view(b,-1,h*w) # B X C X N
        proj_value_from_text = self.value_from_text(text_feat).view(b*c, -1)

        img_attentioned = torch.bmm(proj_value_from_img, attention.permute(0,2,1)).view(b*c, -1)# .view(b*c, -1)

        matched_feature = F.cosine_similarity(img_attentioned, proj_value_from_text)
        matched_feature = matched_feature.view(b, c)
        return matched_feature    

    def forward(self, img_feat, text_feat):
        b, c, h, w = img_feat.shape

        proj_query  = self.query_from_text(text_feat).view(b,-1,h*w).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_from_img(img_feat).view(b,-1,h*w) # B X C x (N)   
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        
        proj_value_from_img = self.value_from_img(img_feat).view(b,-1,h*w) # B X C X N
        proj_value_from_text = self.value_from_text(text_feat).view(b, -1) #.view(b*c, -1)

        img_attentioned = torch.bmm(proj_value_from_img, attention.permute(0,2,1)).view(b, -1)# .view(b*c, -1)

        matched_feature = F.cosine_similarity(img_attentioned, proj_value_from_text)
        #matched_feature = matched_feature.view(b, c)
        return matched_feature

class SentenceRecevier(nn.Module):
    def __init__(self, image_feat_channel=256, text_feat_channel=128, kernel_size=32):
        super().__init__()

        self.channel_weights = nn.Parameter(torch.ones(1, image_feat_channel, 2))

    def forward(self, img_feat, text_feat):
        weights = torch.softmax(self.channel_weights, dim=-1)
        weights_img = weights[:,:,0].view(1, -1, 1, 1)
        weight_txt = weights[:,:,1].view(1, -1, 1, 1) + 0.5
        
        return weights_img * img_feat + weight_txt * text_feat


class WordAttention(nn.Module):
    def __init__(self, image_feat_channel=256, kernel_size=32, word_length=15, word_emb=768):
        super().__init__()

        self.word_length = word_length
        self.word_emb = word_emb
        self.kernel_size = kernel_size
        self.image_feat_channel = image_feat_channel

        self.key_from_img = nn.Conv2d(image_feat_channel, image_feat_channel//32, kernel_size=1)

        self.query_from_text = nn.Sequential(UnFlatten(1),
                        nn.ConvTranspose2d(word_emb, image_feat_channel//32, 4, 1, 0),
                        nn.BatchNorm2d(image_feat_channel//32), nn.ReLU(),
                        nn.ConvTranspose2d(image_feat_channel//32, image_feat_channel//32, 4, 2, 1),
                        nn.AdaptiveAvgPool2d(kernel_size))
        
        self.value_from_img = nn.Conv2d(image_feat_channel, image_feat_channel, kernel_size=1)

        self.value_from_text = nn.Sequential(UnFlatten(1),
                        nn.ConvTranspose2d(word_emb, image_feat_channel//4, 4, 1, 0),
                        nn.BatchNorm2d(image_feat_channel//4), nn.ReLU(),
                        nn.ConvTranspose2d(image_feat_channel//4, image_feat_channel, 4, 2, 1),
                        nn.AdaptiveAvgPool2d(kernel_size))

        self.softmax  = nn.Softmax(dim=-1)

    def read_feat(self, word_seq):
        b=word_seq.shape[0]
        word_seq = word_seq.transpose(1,0).contiguous().view(b*self.word_length, self.word_emb, 1, 1)

        proj_value_from_text = self.value_from_text(word_seq)  # bx15 x c x h x w
        proj_value_from_text = proj_value_from_text.view(self.word_length, b, self.image_feat_channel, self.kernel_size, self.kernel_size).transpose(1,0).contiguous()
        return proj_value_from_text

    def read_img_feat(self, img_feat):
        b, c = img_feat.shape[:2]
        
        proj_key = self.key_from_img(img_feat)
        proj_key = F.adaptive_max_pool2d(proj_key, 1).view(b, c//32) # B X C  
        
        proj_v = self.value_from_img(img_feat)
        proj_v = F.adaptive_max_pool2d(proj_v, 1).view(b, c) # B X C  

        return torch.cat([proj_key, proj_v], dim=1)

    def rf(self, img_feat, word_seq):
        b, c, h, w = img_feat.shape

        # word embedding seq shape: b x 15 x 768
        word_seq = word_seq.transpose(1,0).contiguous().view(b*self.word_length, self.word_emb, 1, 1)

        proj_query  = self.query_from_text(word_seq).view(b*self.word_length,-1,h*w).permute(0,2,1) # (Bx15) X C X (N)
        proj_key =  self.key_from_img(img_feat).view(b,-1,h*w).repeat(self.word_length, 1, 1).contiguous() # (Bx15) X C x (N)   
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # (Bx15) X (N) X (N)           so far we get 15 attention maps
        
        proj_value_from_img = self.value_from_img(img_feat).view(b,-1,h*w).repeat(self.word_length, 1, 1).contiguous() # (Bx15) X C X N
        proj_value_from_text = self.value_from_text(word_seq).view(b*self.word_length*c, -1)  # (Bx15xC) x N 

        img_attentioned = torch.bmm(proj_value_from_img, attention.permute(0,2,1))
        
        proj_value_from_text = proj_value_from_text.view(b*self.word_length*c, -1)
        img_attentioned = img_attentioned.view(b*self.word_length*c, -1)

        matched_feature = F.cosine_similarity(img_attentioned, proj_value_from_text)
        matched_feature = matched_feature.view(self.word_length, b, c).transpose(1,0).contiguous()  # shape: b x 15 x c
        return matched_feature 

    def forward(self, img_feat, word_seq):
        b, c, h, w = img_feat.shape

        # word embedding seq shape: b x 15 x 768
        word_seq = word_seq.transpose(1,0).contiguous().view(b*self.word_length, self.word_emb, 1, 1)

        proj_query  = self.query_from_text(word_seq).view(b*self.word_length,-1,h*w).permute(0,2,1) # (Bx15) X C X (N)
        proj_key =  self.key_from_img(img_feat).view(b,-1,h*w).repeat(self.word_length, 1, 1).contiguous() # (Bx15) X C x (N)   
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # (Bx15) X (N) X (N)           so far we get 15 attention maps
        
        proj_value_from_img = self.value_from_img(img_feat).view(b,-1,h*w).repeat(self.word_length, 1, 1).contiguous() # (Bx15) X C X N
        proj_value_from_text = self.value_from_text(word_seq).view(b*self.word_length*c, -1)  # (Bx15xC) x N 

        img_attentioned = torch.bmm(proj_value_from_img, attention.permute(0,2,1))
        
        proj_value_from_text = proj_value_from_text.view(b*self.word_length*c, -1).view(b, -1)
        img_attentioned = img_attentioned.view(b, -1) #view(b*self.word_length*c, -1)

        matched_feature = F.cosine_similarity(img_attentioned, proj_value_from_text)
        #matched_feature = matched_feature.view(self.word_length, b, c).transpose(1,0).contiguous()  # shape: b x 15 x c
        return matched_feature   # shape: b x 15 x c


class WordRecevier(nn.Module):
    def __init__(self, image_feat_channel=256, text_feat_channel=128, kernel_size=32, word_seq=15):
        super().__init__()

        self.image_feat_channel = image_feat_channel
        self.word_seq = word_seq
        self.channel_weights = nn.Parameter(torch.ones(1, word_seq, image_feat_channel, 2))

    def forward(self, img_feat, text_feat):
        b, c, h, w = img_feat.shape
        ## text feat shape: b x 15 x c x h x w
        weights = torch.softmax(self.channel_weights, dim=-1)
        weights_img = weights[:,:,:,0].view(1, self.word_seq, self.image_feat_channel, 1, 1)
        weight_txt = weights[:,:,:,1].view(1, self.word_seq, self.image_feat_channel, 1, 1) + 0.5
        
        img_feat = img_feat.unsqueeze(1).repeat(1, self.word_seq, 1, 1, 1)
        
        #print(img_feat.shape, weight_txt.shape, weights_img.shape)
        result = weights_img * img_feat + weight_txt * text_feat
        result = result.sum(1) / self.word_seq
        return result

class G(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_4 = nn.Sequential(UnFlatten(1),
                        spectral_norm(nn.ConvTranspose2d(code_dim, channel, 4, 1, 0)),
                        nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_8 = ResBlock(channel)
        self.conv_16 = ResBlock(channel)
        self.conv_32 = ResBlock(channel)
        self.conv_64 = ResBlock(channel)
        self.conv_128 = ResBlock(channel)
        #self.conv_256 = ResBlock(channel, )

        self.to_rgb_64 = nn.Sequential(spectral_norm(nn.Conv2d(channel, 3, 3, 1, 1)), nn.Tanh())
        self.to_rgb_128 = nn.Sequential(spectral_norm(nn.Conv2d(channel, 3, 3, 1, 1)), nn.Tanh())
        #self.to_rgb_256 = nn.Sequential(spectral_norm(nn.Conv2d(channel, 3, 3, 1, 1)), nn.Tanh())

        self.sentence_attn_4  = SentenceAttention(kernel_size=4)
        self.sentence_attn_8  = SentenceAttention(kernel_size=8)
        self.sentence_attn_16 = SentenceAttention(kernel_size=16)

        self.sentence_rec_4 = SentenceRecevier(kernel_size=4)
        self.sentence_rec_8 = SentenceRecevier(kernel_size=8)
        self.sentence_rec_16 = SentenceRecevier(kernel_size=16)

        self.word_attn_4  = WordAttention(kernel_size=4)
        self.word_attn_8  = WordAttention(kernel_size=8)
        self.word_attn_16 = WordAttention(kernel_size=16)

        self.word_rec_4 = WordRecevier(kernel_size=4)
        self.word_rec_8 = WordRecevier(kernel_size=8)
        self.word_rec_16 = WordRecevier(kernel_size=16)

    def forward(self, noise, text_latent, text_emb_seq, step=0):
        feat_4 = self.conv_4(noise)
        
        sentence_feat_4 = self.sentence_attn_4.read_feat(text_latent)
        feat_4 = self.sentence_rec_4(feat_4, sentence_feat_4)
        
        word_feat_4 = self.word_attn_4.read_feat(text_emb_seq)
        feat_4 = self.word_rec_4(feat_4, word_feat_4)

        feat_8 = self.conv_8(feat_4)
        
        sentence_feat_8 = self.sentence_attn_8.read_feat(text_latent)
        feat_8 = self.sentence_rec_8(feat_8, sentence_feat_8)
        
        word_feat_8 = self.word_attn_8.read_feat(text_emb_seq)
        feat_8 = self.word_rec_8(feat_8, word_feat_8)

        feat_16 = self.conv_16(feat_8)

        sentence_feat_16 = self.sentence_attn_16.read_feat(text_latent)
        feat_16 = self.sentence_rec_16(feat_16, sentence_feat_16)
        
        word_feat_16 = self.word_attn_16.read_feat(text_emb_seq)
        feat_16 = self.word_rec_16(feat_16, word_feat_16)


        feat_32 = self.conv_32(feat_16)
        feat_64 = self.conv_64(feat_32)

        if step==0:
            return self.to_rgb_64(feat_64)
        elif step==1:
            feat_128 = self.conv_128(feat_64)
            return self.to_rgb_128(feat_128)


class D(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.d_64 = D_64(channel=channel, code_dim=code_dim)
        self.d_128 = D_128(channel=channel, code_dim=code_dim)
        #self.d_256 = D_256(channel=channel, code_dim=code_dim)
    def forward(self, img, step=0):
        if step==0:
            out = self.d_64(img).view(-1)
        elif step==1:
            out = self.d_128(img).view(-1)
        return out

class D_64(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_32 = nn.Sequential( 
                    spectral_norm(nn.Conv2d(3, channel, 4, 2, 1)),
                    nn.BatchNorm2d(channel))

        self.conv_16 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        self.conv_8 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        self.conv_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))

        self.rf = spectral_norm(nn.Conv2d(channel, 1, 4, 1, 0))

    def forward(self, img):
        feat_32 = self.conv_32(img)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        feat_4 = self.conv_4(feat_8)
        
        return self.rf(feat_4).view(-1)

class D_128(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_64 = nn.Sequential(
                    spectral_norm(nn.Conv2d(3, channel, 4, 2, 1)),
                    nn.BatchNorm2d(channel))

        self.conv_32 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_16 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_8 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_4 = spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, img):
        feat_64 = self.conv_64(img)
        feat_32 = self.conv_32(feat_64)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        feat_4 = self.conv_4(feat_8)
        return feat_4.view(-1)

class D_256(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_128 = nn.Sequential(
                    spectral_norm(nn.Conv2d(3, channel, 4, 2, 1)),
                    nn.BatchNorm2d(channel))
        self.conv_64 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_32 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_16 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_8 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        #self.conv_4 = nn.Sequential(
        #    spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
        #    nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))


        self.rf = spectral_norm(nn.Conv2d(channel, 1, 3, 1, 0))

    def forward(self, img):
        feat_128 = self.conv_128(img)
        feat_64 = self.conv_64(feat_128)
        feat_32 = self.conv_32(feat_64)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        #feat_4 = self.conv_4(feat_8)
        #feat_4, pred_code = self.da(feat_4)
        return self.rf(feat_8).view(-1)#, 0



class D_with_t(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.d_64 = D_64_with_t(channel=channel, code_dim=code_dim)
        self.d_128 = D_128_with_t(channel=channel, code_dim=code_dim)
        #self.d_256 = D_256(channel=channel, code_dim=code_dim)
    def forward(self, img, text_latent, text_emb_seq, step=0):
        if step==0:
            out_sent, out_word = self.d_64(img, text_latent, text_emb_seq)
        elif step==1:
            out_sent, out_word = self.d_128(img, text_latent, text_emb_seq)
        return out_sent.view(-1), out_word.view(-1)

class D_64_with_t(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_32 = nn.Sequential(
                    spectral_norm(nn.Conv2d(3, channel, 4, 2, 1)),
                    nn.BatchNorm2d(channel))
        self.conv_16 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_8 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))

        self.sentence_attn_4 = SentenceAttention(kernel_size=4)
        self.sentence_attn_16 = SentenceAttention(kernel_size=16)

        self.word_attn_4  = WordAttention(kernel_size=4)
        self.word_attn_16 = WordAttention(kernel_size=16)

        self.rf_sentence = spectral_norm(nn.Linear(channel*3, 1))
        self.rf_word = spectral_norm(nn.Linear(channel*3, 1))

    def forward(self, img, text_latent, text_emb_seq):
        feat_32 = self.conv_32(img)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        feat_4 = self.conv_4(feat_8)

        sent_feat_4 = self.sentence_attn_4.rf(feat_4, text_latent)
        sent_feat_16 = self.sentence_attn_16.rf(feat_16, text_latent)
        
        word_feat_4 = self.word_attn_4.rf(feat_4, text_emb_seq)
        word_feat_16 = self.word_attn_16.rf(feat_16, text_emb_seq)
        b, r = word_feat_16.shape[:2]

        feat_4 = F.adaptive_avg_pool2d(feat_4, 1).squeeze(-1).squeeze(-1)

        rf_sent = self.rf_sentence(torch.cat([feat_4, sent_feat_4, sent_feat_16], dim=1)).view(-1)

        rf_word = self.rf_word(torch.cat([feat_4.unsqueeze(1).repeat(1,r,1), word_feat_4, word_feat_16], dim=1).view(b*r,-1))

        return rf_sent, rf_word

class D_128_with_t(nn.Module):
    def __init__(self, channel=256, code_dim=128):
        super().__init__()

        self.conv_64 = nn.Sequential(
                    spectral_norm(nn.Conv2d(3, channel, 4, 2, 1)),
                    nn.BatchNorm2d(channel))

        self.conv_32 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_16 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_8 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        self.conv_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))

        self.sentence_attn_4 = SentenceAttention(kernel_size=4)
        self.sentence_attn_16 = SentenceAttention(kernel_size=16)

        self.word_attn_4  = WordAttention(kernel_size=4)
        self.word_attn_16 = WordAttention(kernel_size=16)

        self.rf_sentence = spectral_norm(nn.Linear(channel*3, 1))
        self.rf_word = spectral_norm(nn.Linear(channel*3, 1))

    def forward(self, img, text_latent, text_emb_seq):
        feat_64 = self.conv_64(img)
        feat_32 = self.conv_32(feat_64)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        feat_4 = self.conv_4(feat_8)

        sent_feat_4 = self.sentence_attn_4.rf(feat_4, text_latent)
        sent_feat_16 = self.sentence_attn_16.rf(feat_16, text_latent)
        
        word_feat_4 = self.word_attn_4.rf(feat_4, text_emb_seq)
        word_feat_16 = self.word_attn_16.rf(feat_16, text_emb_seq)
        b, r = word_feat_16.shape[:2]

        feat_4 = F.adaptive_avg_pool2d(feat_4, 1).squeeze(-1).squeeze(-1)

        rf_sent = self.rf_sentence(torch.cat([feat_4, sent_feat_4, sent_feat_16], dim=1)).view(-1)

        rf_word = self.rf_word(torch.cat([feat_4.unsqueeze(1).repeat(1,r,1), word_feat_4, word_feat_16], dim=1).view(b*r,-1))

        return rf_sent, rf_word

class Text_memory(nn.Module):
    def __init__(self, feature_groups=64, image_feat_channel=256, text_feat_channel=128):
        super().__init__()

        self.ifc = image_feat_channel
        self.fg = feature_groups
        ## this module will generate 10 feature-maps that considered as the template features
        ## from the text latent code, the current feature size is 8
        ## the feature this part generates will be applied on the image feature in a convolution way
        self.feat_from_text = nn.Sequential(UnFlatten(1),
                        nn.ConvTranspose2d(text_feat_channel, feature_groups, 4, 1, 0),
                        nn.BatchNorm2d(feature_groups), nn.ReLU(),
                        nn.ConvTranspose2d(feature_groups, feature_groups, 4, 2, 1),
                        nn.BatchNorm2d(feature_groups), nn.ReLU())
        
    
    def forward(self, img_feat, text_feat):
        b, c, h, w = img_feat.shape

        template_from_text = self.feat_from_text(text_feat) # size: b x group x 8 x 8
        ## expand the template's size by repeating the weights
        template_from_text = template_from_text.unsqueeze(2).repeat(1,1,self.ifc//self.fg,1,1) 
        template_from_text = template_from_text.view(1, b*self.ifc, 8, 8).transpose(0,1).contiguous()

        ## since inside one batch, each image have different conv weights, I use the grouped-conv
        ## operation to achieve the convolution operation
        matched_feature = F.conv2d(img_feat.view(1, b*c, h, w), weight=template_from_text, stride=4, groups=b*c)
        matched_feature = F.adaptive_max_pool2d(matched_feature, 1).view(b, c)

        return F.relu(matched_feature)/64





class ImageAE(nn.Module):
    def __init__(self, channel=256):
        super().__init__()
        
        self.encoder = ImageEncoder(channel=channel)
        self.decoder = ImageDecoder(channel=channel)

    def forward(self, img):
        feat_16, feat_8, feat_4 = self.encoder(img)
        return self.decoder(feat_4), feat_4, feat_8, feat_16

class ImageEncoder(nn.Module):
    def __init__(self, channel=256):
        super().__init__()

        self.conv_64 = nn.Sequential(
                    nn.Conv2d(3, channel, 4, 2, 1),
                    nn.BatchNorm2d(channel))
        
        self.conv_32 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        self.conv_16 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        self.conv_8 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.1))
        
        self.conv_4 = nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, img):
        feat_64 = self.conv_64(img)
        feat_32 = self.conv_32(feat_64)
        feat_16 = self.conv_16(feat_32)
        feat_8 = self.conv_8(feat_16)
        feat_4 = self.conv_4(feat_8)

        return feat_16, feat_8, feat_4


class UpConvBlock(nn.Module):
    def __init__(self, channel, ):
        super().__init__()
        
        self.conv = nn.Sequential(  spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)),
                                    nn.BatchNorm2d(channel), 
                                    nn.ReLU())
    def forward(self, feat):
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
        res = self.conv(feat)
        return feat + res

class ImageDecoder(nn.Module):
    def __init__(self, channel=256):
        super().__init__()

        self.conv_8 = UpConvBlock(channel)
        self.conv_16 = UpConvBlock(channel)
        self.conv_32 = UpConvBlock(channel)
        self.conv_64 = UpConvBlock(channel)
        self.conv_128 = UpConvBlock(channel)

        self.to_rgb_128 = nn.Sequential(spectral_norm(nn.Conv2d(channel, 3, 3, 1, 1)), nn.Tanh())

    def forward(self, feat_4):
        feat_8 = self.conv_8(feat_4)
        feat_16 = self.conv_16(feat_8)
        feat_32 = self.conv_32(feat_16)
        feat_64 = self.conv_64(feat_32)
        feat_128 = self.conv_128(feat_64)
        return self.to_rgb_128(feat_128)

