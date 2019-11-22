import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from pytorch_transformers import BertModel, BertTokenizer
import pickle

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


class Text_Latent_G(nn.Module):
    def __init__(self, latent=128, noise=128):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Linear(noise, latent)), nn.ReLU(), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.ReLU(), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.ReLU(), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.ReLU(),
            spectral_norm(nn.Linear(latent, latent))
        )
    def forward(self, noise):
        return self.main(noise)


class Text_Latent_D(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Linear(latent, latent)), nn.LeakyReLU(0.1), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.LeakyReLU(0.1), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.LeakyReLU(0.1), nn.BatchNorm1d(latent),
            spectral_norm(nn.Linear(latent, latent)), nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(latent, 1))
        )
    def forward(self, latent):
        return self.main(latent).view(-1)


class Text_VAE(nn.Module):
    def __init__(self, meta_data_root, vocab_size=2098, channels=256, latent=128, ae_type=0, seq_len=15, emb_size=768):
        super().__init__()
        
        self.latent_size = latent
        self.vocab_size = vocab_size

        self.ae_type = ae_type # ae_type 0 is deterministic, ae_type 1 is probablistic

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.to_bert_idx = pickle.load((open(meta_data_root+'/bird2bert.pkl', 'rb')))


        encode_latent = latent
        if self.ae_type == 1:
            encode_latent *= 2

        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(1, channels, (3, 32), (1, 16), (1, 8))), # 15x48
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(channels, channels, (3, 12), (1, 6), (0, 0))), # 13x7
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(channels, channels, 3, 2, 1)), # 7x4
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(channels, latent, (3, 3), (2, 1), 0)), # 3x2
            nn.LeakyReLU(0.1),
            Flatten(),
            spectral_norm(nn.Linear(latent*3*2, encode_latent))
        )
        
        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(latent, channels, (7, 4), 1, 0)),
            nn.ReLU(),
            spectral_norm(nn.ConvTranspose2d(channels, channels, 3, 2, 1)),
            nn.ReLU(),
            spectral_norm(nn.ConvTranspose2d(channels, channels, (3, 12), (1, 6), (0, 0))),
            nn.ReLU(),
            spectral_norm(nn.ConvTranspose2d(channels, 1, (3, 32), (1, 16), (1, 8))),  # 15 x 768
        )
        self.decoder_fc = nn.Sequential(
            Flatten(),
            spectral_norm(nn.Linear(768, vocab_size))
        )

    def encode(self, embs):
        #embs = self.bert(bert_idx)[0]
        embs = embs.view(embs.shape[0], 1, embs.shape[-2], embs.shape[-1])
        latent = self.encoder(embs)
        if self.ae_type==0:
            return latent 
        elif self.ae_type==1:
            return latent[:,:self.latent_size], latent[:,self.latent_size:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, latent):
        latent = latent.view(latent.shape[0], latent.shape[1], 1, 1)
        decode_emb = self.decoder(latent)
        logits = self.decoder_fc(decode_emb.view(-1, decode_emb.shape[-1]))
        return logits, decode_emb.view(-1, 15, 768)
    
    def forward(self, embs):
        if self.ae_type==1:
            mu, logvar = self.encode(embs)
            z = self.reparameterize(mu, logvar)
            logits, decode_emb = self.decode(z)
            return logits, decode_emb, mu, logvar
        elif self.ae_type==0:
            z = self.encode(embs)
            logits, decode_emb = self.decode(z)
            return logits, decode_emb, z

    def generate(self, latent):
        b = latent.shape[0]
        logits, _ = self.decode(latent)
        logits = logits.view(b, -1, logits.shape[-1])
        idxs = logits.topk(1, dim=-1)[1]
        idxs = idxs.view(b, -1)
        idxs_numpy = idxs.data.cpu().numpy().astype(int)
        g_txt_list = []
        for idx in idxs_numpy:
            g_txt = ''.join(list(map( lambda i: self.bertTokenizer.decode(self.to_bert_idx.get(i)).replace(' ', '')+' ', idx)))   
            g_txt_list.append(g_txt)

        return g_txt_list