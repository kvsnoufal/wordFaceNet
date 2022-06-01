import pandas as pd
from config import Config as cfg
import numpy as np

def get_all_chars(dftrain,dfeval):
    all_chars = set()
    dftrain["context"].apply(lambda x: all_chars.update(list(x)))
    dfeval["context"].apply(lambda x: all_chars.update(list(x)))
    all_chars = list(all_chars)
    all_chars.extend(["[CLS]","[SEP]","<PAD>"])
    token_mapping = {t:i for i,t in enumerate(all_chars)}
    return all_chars,token_mapping

def tokenize(word,context,token_mapping):
    input_tokens = ["[CLS]"]
    input_tokens.extend(list(word))
    input_tokens.append("[SEP]")
    token_type_ids = [0]*len(input_tokens)
    remaining_spaces = cfg.MAX_LEN -len(input_tokens)-1
    input_tokens.extend(list(context)[:remaining_spaces])
    input_tokens.append("[SEP]")
    token_type_ids.extend([1]*(len(list(context)[:remaining_spaces])+1))
    attention_mask = [0]*len(token_type_ids)
    padding_spaces = cfg.MAX_LEN - len(input_tokens)
    if padding_spaces>0:
        padding=["<PAD>"]*padding_spaces
        input_tokens.extend(padding)
        token_type_ids.extend([0]*padding_spaces)
        attention_mask.extend([1]*padding_spaces)
        
    input_token_ids = [token_mapping[t] for t in input_tokens]
    return input_tokens,input_token_ids,token_type_ids,attention_mask    
def get_all_words_contexts(triplet_ds):
    df_n = triplet_ds[["index","right_word","orig_context"]]
    df_n.columns = ["index","word","context"]
    df_n_ = triplet_ds[["index","wrong_word","context"]]
    df_n_.columns = ["index","word","context"]
    df_n = pd.concat([df_n,df_n_],ignore_index=True)
    return df_n
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)    