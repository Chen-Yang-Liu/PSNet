import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_

from torch import Tensor
from typing import Optional, Any, Union, Callable
import cv2
# from cv2 import resize
import numpy as np
from torch.nn import functional as F
from imageio import imread
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.modules.container import ModuleList
import copy

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class Scale_aware(nn.Module):
    def __init__(self, dropout, d_model=1024, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(Scale_aware, self).__init__()
        self.attention_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.linear3 = nn.Linear(d_model, d_model * 4)
        self.linear4 = nn.Linear(d_model * 4, d_model)


    def forward(self, input_q, input_k1, input_k2):
        # FIXME:Scale_aware_transformer_vit_b_32_v2
        input_q_norm = self.norm1(input_q)
        input_k1_norm = self.norm1(input_k1)
        input_k2_norm = self.norm1(input_k2)
        input_k1_norm = torch.cat([input_k1_norm,input_k2_norm],dim=0)
        attn_output_1, attn_weight_1 = self.attention_1(input_q_norm, input_k1_norm, input_k1_norm)  # (Q,K,V)
        output = input_q
        output_norm = self.norm2(attn_output_1)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output_norm))))
        output = output + self.dropout3(ff_output)
        return output


class Cross_att(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(Cross_att, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input, dif):
        # 第一种 RSICCformer_D (diff_as_kv)

        input_norm, dif_norm = self.norm1(input), self.norm1(dif)
        attn_output, attn_weight = self.attention(input_norm, dif_norm, dif_norm)  # (Q,K,V)
        output = input + self.dropout1(attn_output)

        output_norm = self.norm2(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output_norm))))
        output = output + self.dropout3(ff_output)
        return output

class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        # self.Cross_att_concat = Cross_att(dropout, d_model, n_head)
        d_model = d_model*2
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.linear3 = nn.Linear(d_model, 512)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        batch_size = input1.size()[1]

        # 消融实验baseline
        # output_1 = self.cross(input1, input1)  # (Q,K,V)
        # output_2 = self.cross(input2, input2)  # (Q,K,V)

        # 改进dif_as_kv
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)
        output_1 = self.linear3(output_1)#[..., :512]
        output_2 = self.linear3(output_2)#[..., :512]

        # 改进dif_as_kv_2
        # batch = input1.size(1)
        # h = int(pow(input1.size(0),0.5))
        # dif = (input2 - input1)
        # input1_dif = torch.cat([input1,dif],dim=-1).permute(1, 2, 0).view(batch, self.d_model * 2, h, h)
        # input2_dif = torch.cat([input2,dif],dim=-1).permute(1, 2, 0).view(batch, self.d_model * 2, h, h)
        # input1_dif = self.conv_1(input1_dif).view(batch, self.d_model, -1).permute(2, 0, 1)
        # input2_dif = self.conv_2(input2_dif).view(batch, self.d_model, -1).permute(2, 0, 1)
        # output_1 = self.Cross_att(input1, input1_dif)  # (Q,K,V)
        # output_2 = self.Cross_att(input2, input2_dif)  # (Q,K,V)

        return output_1, output_2

    def cross(self, input, dif):
        # 第一种 RSICCformer_D (diff_as_kv)
        input = torch.cat([input, dif], dim=-1)

        input_norm = self.norm1(input)
        # input_norm, dif_norm = self.norm1(input), self.norm1(dif)
        attn_output, attn_weight = self.attention(input_norm, input_norm, input_norm)  # (Q,K,V)
        output = input + self.dropout1(attn_output)

        output_norm = self.norm2(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output_norm))))
        output = output + self.dropout3(ff_output)
        return output


class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model

        # n_layers = 2
        print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))

        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)
        self.transformer_cross = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        # self.transformer2 = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

        # FIXME:helpful
        encoder_self_layer = nn.TransformerEncoderLayer(2*d_model, n_head, dim_feedforward=int(4*d_model))
        self.transformer_concatlength = _get_clones(encoder_self_layer, n_layers)
        self.linear_list = nn.ModuleList([nn.Linear(2*d_model,d_model) for i in range(n_layers)])

        self.Scale_aware_list = nn.ModuleList([Scale_aware(dropout, d_model=1024, n_head=8) for i in range(n_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        if feature_dim == 1024:
            img_feat1 = self.projection(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection2(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 512:
            img_feat1 = self.projection3(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection3(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection4(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)

        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)  # (batch, d_model, h, w)

        img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
        img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

        encoder_output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        encoder_output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

        output1 = encoder_output1
        output2 = encoder_output2
        output1_list = list()
        output2_list = list()
        dif_list = list()
        # layer_list = list()
        for k in range(self.n_layers):
            output1 = torch.cat([output1, output2-output1], dim=-1)
            output1 = self.transformer_concatlength[k](output1)
            output1 = self.linear_list[k](output1)

            output2 = torch.cat([output2, output2 - output1], dim=-1)
            output2 = self.transformer_concatlength[k](output2)
            output2 = self.linear_list[k](output2)

            output1_list.append(output1)
            output2_list.append(output2)

        output_layer1 = torch.cat([output1_list[0], output2_list[0]], dim=-1)
        output_layer2 = torch.cat([output1_list[1], output2_list[1]], dim=-1)
        output_layer3 = torch.cat([output1_list[2], output2_list[2]], dim=-1)

        # Scale_aware
        output_layer1_2 = self.Scale_aware_list[0](output_layer1,output_layer2,output_layer3)
        output_layer2_2 = self.Scale_aware_list[1](output_layer2,output_layer1,output_layer3)
        output_layer3_2 = self.Scale_aware_list[2](output_layer3,output_layer1,output_layer2)


        output = torch.cat([output_layer1_2.unsqueeze(-1), output_layer2_2.unsqueeze(-1), output_layer3_2.unsqueeze(-1)],
                           dim=-1)
        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        # self.fc = nn.Linear(d_model, 2)
        self.fc = nn.Linear(d_model, d_model)
        # self.conv = nn.Conv2d(d_model, d_model, kernel_size=14, stride=14, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=(14, 14), stride=(14, 14))
        self.d_model = d_model
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model,num_layers=1)
        self.decode_step = nn.LSTMCell(d_model, d_model, bias=True)


        self.init_weights()


        self.init_h = nn.Linear(d_model, d_model)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(d_model, d_model)

    def init_hidden_state(self, dif):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # dif： N，d
        h = self.init_h(dif)  # (batch_size, decoder_dim)
        c = self.init_c(dif)
        return h, c

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # #
        enc_att, att_weight = self._mha_block2((self_att_tgt),
                                               torch.cat([memory[:, :, :],], dim=0), memory_mask,
                                               memory_key_padding_mask)

        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight
    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x),att_weight
    def _mha_block3(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn3(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x), att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout=0.5):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 3
        print("decoder_n_layers=", n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding

        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = _get_clones(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def caption_decoder(self, tgt_embedding, memory, tgt_mask):
        i = 3
        out = tgt_embedding
        for layer in self.transformer:
            i = i-1
            out = layer(out, memory[:,:,:,i], tgt_mask=tgt_mask)
        pred = out
        return pred

    def forward(self, memory, encoded_captions, caption_lengths):
        """
        :param memory: image feature (S, batch, feature_dim)
        :param tgt: target sequence (length, batch)
        :param sentence_index: sentence index of each token in target sequence (length, batch)
        """
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        pred = self.caption_decoder(tgt_embedding, memory, tgt_mask=mask)  # (length, batch, feature_dim)
        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)

        pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind
