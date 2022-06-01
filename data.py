import torch
import utils
class InterimEvalDataset(torch.utils.data.Dataset):
    def __init__(self,df,token_mapping):
        self.df = df
        self.token_mapping = token_mapping
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        word = self.df.loc[idx,"word"]
        context = self.df.loc[idx,"context"]
        input_tokens,input_token_ids,token_type_ids,attention_mask = utils.tokenize(word,context,self.token_mapping)
        input_token_ids = torch.tensor(input_token_ids,dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask,dtype=torch.long)

        return {"input_token_ids" :input_token_ids,\
                "token_type_ids" :token_type_ids,\
                "attention_mask" :attention_mask}
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self,df,token_mapping):
        self.df = df
        self.token_mapping = token_mapping

    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        anchor_word = self.df.loc[idx,"right_word"]
        anchor_context = self.df.loc[idx,"orig_context"]

        pos_index = self.df.loc[idx,"hard_pos_index"]
        pos_word = self.df.loc[pos_index,"wrong_word"]
        pos_context = self.df.loc[pos_index,"context"]

        neg_index = self.df.loc[idx,"hard_pneg_index"]
        neg_word = self.df.loc[neg_index,"wrong_word"]
        neg_context = self.df.loc[neg_index,"context"]
        # print(idx)
        # print(anchor_word)
        # print(anchor_context)

        _,a_input_token_ids,a_token_type_ids,a_attention_mask = utils.tokenize(anchor_word,anchor_context,self.token_mapping)
        _,p_input_token_ids,p_token_type_ids,p_attention_mask = utils.tokenize(pos_word,pos_context,self.token_mapping)
        _,n_input_token_ids,n_token_type_ids,n_attention_mask = utils.tokenize(neg_word,neg_context,self.token_mapping)

        return {"a_input_token_ids":torch.tensor(a_input_token_ids,dtype=torch.long),
                "a_token_type_ids":torch.tensor(a_token_type_ids,dtype=torch.long),
                "a_attention_mask":torch.tensor(a_attention_mask,dtype=torch.long),
                "p_input_token_ids":torch.tensor(p_input_token_ids,dtype=torch.long),
                "p_token_type_ids":torch.tensor(p_token_type_ids,dtype=torch.long),
                "p_attention_mask":torch.tensor(p_attention_mask,dtype=torch.long),
                "n_input_token_ids":torch.tensor(n_input_token_ids,dtype=torch.long),
                "n_token_type_ids":torch.tensor(n_token_type_ids,dtype=torch.long),
                "n_attention_mask":torch.tensor(n_attention_mask,dtype=torch.long),
                }


        
