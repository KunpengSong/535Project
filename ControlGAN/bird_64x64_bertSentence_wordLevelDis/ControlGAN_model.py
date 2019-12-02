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

'''
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
'''


class Text_bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def get_word_emb(self,words): #tensor: b,15
        embs = self.bert(words)
        word_embs = embs[0] #b,15,768
        mem = embs[1] 
        return word_embs,mem
    
    def get_sentence_emb(self,mem): # use [4,8,10,12] from [0,1,...,12]
        return mem


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
    def __init__(self, channel=512, noise_dim=256, sentence_dim=768, word_dim=768):
        super().__init__()

        self.conv_8 = nn.Sequential(
                        UnFlatten(),
                        nn.ConvTranspose2d(noise_dim, channel*2, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(channel*2), 
                        GLU(),
                        upBlock(channel, channel//2))

        self.attn_8 = Attn(word_dim, 8, channel//2)

        self.conv_32 = nn.Sequential(
                        upBlock(channel, channel//2),
                        upBlock(channel//2, channel//4))

        self.attn_32 = Attn(word_dim, 32, channel//4)
        
        self.conv_64 = upBlock(channel//2, channel//4)
        self.attn_64 = Attn(word_dim, 64, channel//4)        

        #self.conv_128 = nn.Sequential(
        #                upBlock(channel//4, channel//8))

        self.to_rgb_64 = nn.Sequential(conv3x3(channel//4, 3), nn.Tanh())
        #self.to_rgb_128 = nn.Sequential(conv3x3(channel//8, 3), nn.Tanh())

        self.apply(weights_init)


    def forward(self, noise,sentence,word_emb): #b,256; b,256; 4, 15, 768
        #feat = torch.cat([noise,sentence],dim=-1).unsqueeze(-1).unsqueeze(-1) #b,256; b,256 --> b,512,1,1
        feat = noise.unsqueeze(-1).unsqueeze(-1) #b,256; b,256 --> b,512,1,1

        feat_8 = self.conv_8(feat)
        feat_att_8 = self.attn_8(feat_8,word_emb)
        feat_32 = self.conv_32(torch.cat([feat_8,feat_att_8],dim=1)) 
        feat_att_32 = self.attn_32(feat_32, word_emb)
        feat_64 = self.conv_64(torch.cat([feat_32,feat_att_32],dim=1))
        #feat_att_64 = self.attn_64(feat_64, word_emb)
        #feat_128 = self.conv_128(feat_64)
        
        return [self.to_rgb_64(feat_64)]#self.to_rgb_128(feat_128)]


class D(nn.Module):
    def __init__(self, channel=512, sentence_dim = 768, word_dim=768):
        super().__init__()

        #self.from_128 = nn.Sequential(
        #    downBlock(3, channel//8),
        #    downBlock(channel//8, channel//4),
        #    downBlock(channel//4, channel//4),
        #    downBlock(channel//4, channel//2),
        #)

        self.from_64 = nn.Sequential( 
            downBlock(3, channel//8),
            downBlock(channel//8, channel//4),
            downBlock(channel//4, channel//2),
        )

        self.rf_64 = nn.Sequential(
            downBlock(channel//2, channel),
            spectral_norm(nn.Conv2d(channel, 1, 4, 1, 0, bias=True))
        )

        #self.rf_128 = nn.Sequential(
        #    downBlock(channel//2, channel),
        #    spectral_norm(nn.Conv2d(channel, 1, 4, 1, 0, bias=True))
        #)

        #self.it_match = TextImageMatcher(channel//2, word_dim)

        self.matcher_1 = nn.Sequential(
            downBlock(channel//2, channel),
            spectral_norm(nn.Conv2d(channel,channel, 4, 1, 0, bias=True)),
        )

        self.matcher_2 = nn.Sequential(
            nn.Linear(channel+sentence_dim,channel),
            nn.LeakyReLU(0.2),
            nn.Linear(channel,channel//2),
            nn.LeakyReLU(0.2),
            nn.Linear(channel//2,1),
            )
        
        self.wordLevelDis = WordLevelDiscriminator(channel//2,word_dim)

        self.apply(weights_init)


    def forward(self, imgs, sentence=None, word_emb=None, train_perm=True):
        feat_64 = self.from_64(imgs[0]) 
        #feat_128 = self.from_128(imgs[1])

        logit_1 = self.rf_64(feat_64).view(-1)
        #logit_3 = self.rf_128(feat_128).view(-1)
        
        b = imgs[0].shape[0]
        #img_feat = torch.cat([feat_64, feat_128], dim=1)
        img_feat = feat_64

        match = match_perm = match_word = match_perm_word = None

        #match = self.it_match(img_feat, sentence)
        #match = self.matcher_2(torch.cat([self.matcher_1(img_feat), sentence.unsqueeze(-1).unsqueeze(-1)],dim=1).squeeze(-1).squeeze(-1)) #b,256
        match_word = self.wordLevelDis(img_feat,word_emb)

        if train_perm:
            perm = true_randperm(b)
            #match_perm = self.it_match(img_feat[perm], sentence)
            #match_perm = self.matcher_2(torch.cat([self.matcher_1(img_feat[perm]), sentence.unsqueeze(-1).unsqueeze(-1)],dim=1).squeeze(-1).squeeze(-1)) #b,256
            match_perm_word = self.wordLevelDis(img_feat[perm],word_emb)
            
        return torch.cat([logit_1]), match, match_perm, match_word, match_perm_word



class WordLevelDiscriminator(nn.Module):
    def __init__(self, img_channel,word_dim):
        super(WordLevelDiscriminator,self).__init__()
        self.FC = nn.Linear(img_channel,word_dim)
        self.softmax  = nn.Softmax(dim=-1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sig = nn.Sigmoid()

    def forward(self,feat,word_emb): #b,c,h,w; b,l,768
        b,c,h,w = feat.shape
        feat = feat.view(b,c,-1).permute(0,2,1) #b,c,h,w --> b,c,h*w --> b,h*w,c
        feat = self.FC(feat).permute(0,2,1) # b,h*w,c -->  b,h*w,768 --> b,768,h*w

        att = self.softmax(torch.bmm(word_emb,feat)) #b,l,768;b,768,h*w --> b,l,h*w
        att_feat = torch.bmm(feat,att.permute(0,2,1)) #b,768,l

        avg = torch.mean(word_emb,1,True) #b,1,768
        self_word_att = torch.bmm(avg,word_emb.permute(0,2,1)) #b,1,l

        new_feat = torch.mul(att_feat,self_word_att.repeat(1,768,1)) #b,768,l
        cos = torch.mean(self.sig(self.cos(new_feat.permute(0,2,1),word_emb)),dim=-1) #b
        return cos


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

class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb