

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from posencode import PositionEmbeddingSine


class Transformer(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_encoder_layers=6,
                 n_patches=50, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.position_embeddings = nn.Parameter(torch.zeros(n_patches+1, 1, d_model))
        self._reset_parameters()


        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        cls_tokens = self.cls_token.expand(1, bs, -1)
        src = src.flatten(2).permute(2, 0, 1)
        src = torch.cat((cls_tokens, src), dim=0)
        memory,attention_weights = self.encoder(src, pos=self.position_embeddings)
        
        return  memory.permute(1, 2, 0),attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        attention_weights=[]
        for layer in self.layers:
            output ,attention_weight= layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            attention_weights.append(attention_weight)
        if self.norm is not None:
            output = self.norm(output)

        return output,attention_weights




class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2,attention_weight = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attention_weight

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
    

if __name__=='__main__':
    
    
    d_modelt=256
    nheadt=8
    num_encoder_layerst=2
    dim_feedforwardt=1024
    dropout=0.1
    normalize_beforet=False
        
        
    transformer = Transformer(d_model=d_modelt,nhead=nheadt,num_encoder_layers=num_encoder_layerst,
    dim_feedforward=dim_feedforwardt,normalize_before=normalize_beforet)
    
    hidden_dim = d_modelt
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    src = torch.rand(1, hidden_dim, 5, 5)
    pos_add = pos_enc(src)

    out = transformer(src,pos_embed = pos_add)
    print(torch.sum(out),out.shape)
