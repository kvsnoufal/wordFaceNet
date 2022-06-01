import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from torch import Tensor
import math
import torch.nn.functional as F
from tqdm import tqdm
from copy import copy
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import time

from config import Config as cfg
import utils


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Embedder(nn.Module):
    def __init__(self,embed_dim=200,token_mapping=None):
        super(Embedder,self).__init__()
        self.token_embed = nn.Embedding(len(token_mapping),embed_dim)
        self.token_type_embed = nn.Embedding(2,embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim,0.1,cfg.MAX_LEN)
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
    def forward(self,input_token,token_type):
        embedding = self.token_embed(input_token) + self.token_type_embed(token_type)
        embedding = self.layerNorm(embedding) * math.sqrt(self.embed_dim)
        
        embedding = self.pos_encoder(embedding.permute(1,0,2)).permute(1,0,2)
        return embedding        
class TransformerModel(nn.Module):
    def __init__(self,embed_dim=200,nhead=4,d_hid=256,dropout=0.2,nlayers=2,token_mapping=None):
        super(TransformerModel,self).__init__()
        self.embeddingLayer = Embedder(embed_dim,token_mapping)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier = nn.Linear(embed_dim*cfg.MAX_LEN,cfg.OUTPUT_DIM)
    def forward(self,x,mask,token_type_inp):
        x = self.embeddingLayer(x,token_type_inp)
        # print(x.shape,mask.shape)
        output = self.transformer_encoder(src=x,src_key_padding_mask=mask)
        output = nn.Flatten()(output)
        # print(output.shape)
        output = self.classifier(output)
        output = F.normalize(output,p=2)
        # print(output.shape)
        return output        