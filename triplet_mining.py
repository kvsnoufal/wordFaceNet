import pandas as pd
from tqdm import tqdm
import torch
from config import Config as cfg
from utils import softmax
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import os


def get_mined_df(interdl,trf,triplet_ds,epoch,train=True):
    if cfg.DEBUG:
        triplet_ds["hard_pos_index"]= 0
        triplet_ds["hard_pneg_index"]= 0
        return triplet_ds
        
    all_word_vecs = []
    for i, batch in tqdm(enumerate(interdl),total=len(interdl)):
        input_token_ids = batch["input_token_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        input_token_ids = input_token_ids.to(cfg.DEVICE)
        token_type_ids = token_type_ids.to(cfg.DEVICE)
        attention_mask = attention_mask.to(cfg.DEVICE)
        with torch.no_grad():
            op = trf(input_token_ids,attention_mask,token_type_ids)
            all_word_vecs.extend(list(op.detach().cpu().numpy()))
    all_word_vecs= np.array(all_word_vecs)
    all_indices = list(range(len(triplet_ds)))
    for i,row in tqdm(triplet_ds.iterrows(),total=len(triplet_ds)):
        anchor_vec = all_word_vecs[row["index"]].reshape(1,-1)
        pos_indices = row["pos_indices"]
        pos_word_vecs = all_word_vecs[triplet_ds.shape[0]:]
        pos_word_vecs = pos_word_vecs[pos_indices]
        
        pos_dists = euclidean_distances(anchor_vec,pos_word_vecs).flatten()
        pos_p = softmax(pos_dists)
        hard_pos_index = np.random.choice(pos_indices,1,replace=False,p=pos_p)[0]

        sampled_word_vecs = all_word_vecs[triplet_ds.shape[0]:]
        
        sampled_indices = np.random.choice(np.arange(len(sampled_word_vecs)),1000)
        mask = ~np.isin(sampled_indices,pos_indices)
        sampled_indices = sampled_indices[mask]

        sampled_word_vecs = sampled_word_vecs[sampled_indices]
        neg_dists = euclidean_distances(anchor_vec,sampled_word_vecs).flatten()
        neg_dists = 0 - neg_dists
        pos_n = softmax(neg_dists)
        hard_neg_index = np.random.choice(sampled_indices,1,replace=False,p=pos_n)[0]
    
        triplet_ds.loc[i,"hard_pos_index"] = hard_pos_index
        triplet_ds.loc[i,"hard_pneg_index"] = hard_neg_index
    folderpath = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.FILE_DIR)   
    if train:     
        filepath = os.path.join(folderpath,"triplet_ds_mined_{}.csv".format(epoch))
    else:
        filepath = os.path.join(folderpath,"triplet_ds_eval_mined_{}.csv".format(epoch))
    triplet_ds.to_csv(filepath,index=None)
    return triplet_ds