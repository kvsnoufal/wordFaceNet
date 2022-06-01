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
import faiss
from torch.utils.tensorboard import SummaryWriter
from config import Config as cfg
import utils
import data
import models
import triplet_mining as tm
from loss_fn  import TripletLoss
from optimizer import WarmupOptimizer

if __name__ =="__main__":
    folderpath = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.LOG_DIR)
    os.makedirs(folderpath,exist_ok=True)
    folderpath = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.FILE_DIR)
    os.makedirs(folderpath,exist_ok=True)
    folderpath = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.MODEL_DIR)
    os.makedirs(folderpath,exist_ok=True)
    

    dftrain  = pd.read_csv(cfg.INPUT_FILE_TRAIN).iloc[:,1:]
    dfeval = pd.read_csv(cfg.INPUT_FILE_EVAL).iloc[:,1:]
    dftrain = dftrain[dftrain["context"].notnull()].reset_index(drop=True)
    dfeval = dfeval[dfeval["context"].notnull()].reset_index(drop=True)
    dftrain = dftrain[dftrain["orig_context"].notnull()].reset_index(drop=True)
    dfeval = dfeval[dfeval["orig_context"].notnull()].reset_index(drop=True)

    all_chars,token_mapping = utils.get_all_chars(dftrain,dfeval)

    
    print("All characters: {}".format(all_chars))
    print("Num tokens ; {}".format(len(token_mapping)))
    
    triplet_ds = dftrain[dftrain["wrong_word"].notnull()].reset_index(drop=True).reset_index()
    print(triplet_ds.shape,triplet_ds.columns)
    df_all_words_contexts = utils.get_all_words_contexts(triplet_ds)
    pos_mapper = triplet_ds.groupby("right_word")["index"].agg(list).to_dict()
    triplet_ds["pos_indices"] = triplet_ds["right_word"].map(pos_mapper)
    triplet_ds["num_pos_indices"] = triplet_ds["pos_indices"].apply(len)


    triplet_ds_eval = dfeval[dfeval["wrong_word"].notnull()].reset_index(drop=True).reset_index()
    print(triplet_ds_eval.shape,triplet_ds_eval.columns)
    df_all_words_contexts_eval = utils.get_all_words_contexts(triplet_ds_eval)
    pos_mapper_eval = triplet_ds_eval.groupby("right_word")["index"].agg(list).to_dict()
    triplet_ds_eval["pos_indices"] = triplet_ds_eval["right_word"].map(pos_mapper_eval)
    triplet_ds_eval["num_pos_indices"] = triplet_ds_eval["pos_indices"].apply(len)

    # for accuracy eval
    right_words_df = dftrain[["right_word","orig_context"]].drop_duplicates()
    right_words_df.columns = ["word","context"]
    right_words_df = right_words_df.reset_index(drop=True)
    
    interdsW = data.InterimEvalDataset(right_words_df,token_mapping)
    interdlW = DataLoader(interdsW,\
                            batch_size=cfg.INTER_EVAL_BATCH_SIZE,\
                            shuffle=False)
    # triplet_mining_dataset

    interds = data.InterimEvalDataset(df_all_words_contexts,token_mapping)
    interdl = DataLoader(interds,\
                            batch_size=cfg.INTER_EVAL_BATCH_SIZE,\
                            shuffle=False)
    interds_eval = data.InterimEvalDataset(df_all_words_contexts_eval,token_mapping)
    interdl_eval = DataLoader(interds_eval,\
                            batch_size=cfg.INTER_EVAL_BATCH_SIZE,\
                            shuffle=False)
    encoderModel = models.TransformerModel(cfg.EMBED_DIM,token_mapping=token_mapping)
    encoderModel.to(cfg.DEVICE)

    # base_optimizer = torch.optim.Adam(encoderModel.parameters(),\
    #     lr=1e-3,weight_decay=0.0001,betas=(0.0,0.999))
    # optimizer = WarmupOptimizer(base_optimizer,\
    #     d_model=cfg.EMBED_DIM,scale_factor=1,warmup_steps=10)
    optimizer = torch.optim.Adam(encoderModel.parameters(),\
        lr=4e-4,weight_decay=0.0001,betas=(0.0,0.999))
    criterion = TripletLoss()
    LOGPATH = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.LOG_DIR)
    writer = SummaryWriter(log_dir=LOGPATH)
    global_start_time = time.time()
    for epoch in range(cfg.EPOCHS):
        print(f"INIT EPOCH {epoch}")
        if epoch%cfg.TRIPLET_MINE_EVERY_N_STEPS==0:
            print("init triplet mining")
            mining_start = time.time()
            encoderModel.eval()
            triplet_ds = tm.get_mined_df(interdl,encoderModel,triplet_ds,epoch)
            print("time to mine: {} mins".format(int((time.time()-mining_start)/60)))


            triplet_ds_eval = tm.get_mined_df(interdl_eval,encoderModel,triplet_ds_eval,epoch,False)
            print("time to mine: {} mins".format(int((time.time()-mining_start)/60)))

        train_ds = data.TripletDataset(triplet_ds,token_mapping)
        eval_ds = data.TripletDataset(triplet_ds_eval,token_mapping)

        train_dataloader = DataLoader(train_ds,\
                        batch_size=cfg.BATCH_SIZE,\
                        shuffle=True)
        eval_dataloader = DataLoader(eval_ds,\
                            batch_size=cfg.BATCH_SIZE,\
                            shuffle=False)
        

        encoderModel.train()
        train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

            a_input_token_ids = batch["a_input_token_ids"]
            a_token_type_ids = batch["a_token_type_ids"]
            a_attention_mask = batch["a_attention_mask"]
            p_input_token_ids = batch["p_input_token_ids"]
            p_token_type_ids = batch["p_token_type_ids"]
            p_attention_mask = batch["p_attention_mask"]
            n_input_token_ids = batch["n_input_token_ids"]
            n_token_type_ids = batch["n_token_type_ids"]
            n_attention_mask = batch["n_attention_mask"]

            a_input_token_ids = a_input_token_ids.to(cfg.DEVICE)
            a_token_type_ids = a_token_type_ids.to(cfg.DEVICE)
            a_attention_mask = a_attention_mask.to(cfg.DEVICE)
            p_input_token_ids = p_input_token_ids.to(cfg.DEVICE)
            p_token_type_ids = p_token_type_ids.to(cfg.DEVICE)
            p_attention_mask = p_attention_mask.to(cfg.DEVICE)
            n_input_token_ids = n_input_token_ids.to(cfg.DEVICE)
            n_token_type_ids = n_token_type_ids.to(cfg.DEVICE)
            n_attention_mask = n_attention_mask.to(cfg.DEVICE)

            a_emb = encoderModel(a_input_token_ids,a_attention_mask,a_token_type_ids)
            p_emb = encoderModel(p_input_token_ids,p_attention_mask,p_token_type_ids)
            n_emb = encoderModel(n_input_token_ids,n_attention_mask,n_token_type_ids)

            loss = criterion(a_emb,p_emb,n_emb)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_dataloader)

        encoderModel.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader)):

                a_input_token_ids = batch["a_input_token_ids"]
                a_token_type_ids = batch["a_token_type_ids"]
                a_attention_mask = batch["a_attention_mask"]
                p_input_token_ids = batch["p_input_token_ids"]
                p_token_type_ids = batch["p_token_type_ids"]
                p_attention_mask = batch["p_attention_mask"]
                n_input_token_ids = batch["n_input_token_ids"]
                n_token_type_ids = batch["n_token_type_ids"]
                n_attention_mask = batch["n_attention_mask"]

                a_input_token_ids = a_input_token_ids.to(cfg.DEVICE)
                a_token_type_ids = a_token_type_ids.to(cfg.DEVICE)
                a_attention_mask = a_attention_mask.to(cfg.DEVICE)
                p_input_token_ids = p_input_token_ids.to(cfg.DEVICE)
                p_token_type_ids = p_token_type_ids.to(cfg.DEVICE)
                p_attention_mask = p_attention_mask.to(cfg.DEVICE)
                n_input_token_ids = n_input_token_ids.to(cfg.DEVICE)
                n_token_type_ids = n_token_type_ids.to(cfg.DEVICE)
                n_attention_mask = n_attention_mask.to(cfg.DEVICE)

                a_emb = encoderModel(a_input_token_ids,a_attention_mask,a_token_type_ids)
                p_emb = encoderModel(p_input_token_ids,p_attention_mask,p_token_type_ids)
                n_emb = encoderModel(n_input_token_ids,n_attention_mask,n_token_type_ids)

                loss = criterion(a_emb,p_emb,n_emb)
                val_loss+=loss.item()
        val_loss = val_loss/len(eval_dataloader)
        print(f"Epoch {epoch}:: Train Loss: {train_loss}; Eval Loss: {val_loss}")
        writer.add_scalar("Loss/train",train_loss,epoch)
        writer.add_scalar("Val/train",val_loss,epoch)
        print(f"time for epoch {epoch}: {int((time.time() - global_start_time)/60)}")
        model_save_folder = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.MODEL_DIR)
        model_save_path = os.path.join(model_save_folder,f"model_{epoch}.pth")
        torch.save(encoderModel.state_dict(), model_save_path)
        print("model Saved")
    

        if (epoch+1)%cfg.ACCURACY_CHECK_FREQ ==0:
            processedEval = dfeval[["wrong_word","context","right_word"]].dropna()
            processedEval.columns = ["word","context","target"]
            processedEval = processedEval.drop_duplicates()
            tempdf = dfeval[["right_word","orig_context"]]
            tempdf.columns=["word","context"]
            tempdf["target"] = tempdf["word"]
            processedEval = pd.concat([processedEval,tempdf],ignore_index=True)\
                .drop_duplicates()\
                    .reset_index(drop=True)
            # print(processedEval.shape)
            # print(processedEval.isnull().sum())
            

            interdsAccEval = data.InterimEvalDataset(processedEval,token_mapping)
            interdlAccEval = DataLoader(interdsAccEval,\
                                    batch_size=cfg.INTER_EVAL_BATCH_SIZE,\
                                    shuffle=False)

            all_word_vecs = []
            
            for i_, batch in tqdm(enumerate(interdlW),total=len(interdlW)):
                input_token_ids = batch["input_token_ids"]
                token_type_ids = batch["token_type_ids"]
                attention_mask = batch["attention_mask"]
                input_token_ids = input_token_ids.to(cfg.DEVICE)
                token_type_ids = token_type_ids.to(cfg.DEVICE)
                attention_mask = attention_mask.to(cfg.DEVICE)
                with torch.no_grad():
                    op = encoderModel(input_token_ids,attention_mask,token_type_ids)
                    all_word_vecs.extend(list(op.detach().cpu().numpy()))
            all_word_vecs = np.array(all_word_vecs)  


            queryEmbeddings = []
            for i_, batch in tqdm(enumerate(interdlAccEval),total=len(interdlAccEval)):
                input_token_ids = batch["input_token_ids"]
                token_type_ids = batch["token_type_ids"]
                attention_mask = batch["attention_mask"]
                input_token_ids = input_token_ids.to(cfg.DEVICE)
                token_type_ids = token_type_ids.to(cfg.DEVICE)
                attention_mask = attention_mask.to(cfg.DEVICE)
                with torch.no_grad():
                    op = encoderModel(input_token_ids,attention_mask,token_type_ids)
                    queryEmbeddings.extend(list(op.detach().cpu().numpy()))
            queryEmbeddings = np.array(queryEmbeddings)  

            res = faiss.StandardGpuResources()  # use a single GPU
            index_flat = faiss.IndexFlatL2(cfg.OUTPUT_DIM)  # build a flat (CPU) index
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(all_word_vecs)         # add vectors to the index
            # print(gpu_index_flat.ntotal)
            k=1
            D, I = gpu_index_flat.search(queryEmbeddings, k)

            processedEval["matched_indx"] = I.flatten()
            processedEval["matched_word"] = processedEval["matched_indx"].map(right_words_df["word"].to_dict())
            processedEval["matched_dist"] = D.flatten()
            processedEval["score"]=0
            processedEval.loc[processedEval["target"]==processedEval["matched_word"],"score"]=1
            print("Accuracy Scores:")
            print(processedEval["score"].value_counts().to_dict())
            print(processedEval["score"].value_counts(normalize=True).to_dict())
            writer.add_scalar("Accuracy",processedEval["score"].value_counts(normalize=True).to_dict()[1],epoch)
            file_save_folder = os.path.join(cfg.OUTPUT_DIR,cfg.RUN_ID,cfg.FILE_DIR)
            file_save_path = os.path.join(file_save_folder,f"processedEval_{epoch}.csv")
            processedEval.to_csv(file_save_path,index=None)