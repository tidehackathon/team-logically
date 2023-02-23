#fever_scorer_dir = '/home/pimverschuuren/code/veracityai/data_processing/'

import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os, glob
from typing import List, Tuple
import requests
import json
from tqdm import tqdm
import numpy as np

from random import randint
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#from fever_evidence_scorer import check_predicted_evidence_format, fever_score

# DEVICE="cpu"

if torch.cuda.is_available():
    DEVICE="cuda"

# DATASET = 'end2end_fact_checking_processed_declare_snopes.jsonl'

# df = pd.read_json(DATASET,lines=True)
claim_var = "claim" # "claim_text"
evid_var = "content" # "evidence_texts" # Politifact/Snopes
label_var = "original_label" # Snopes/Poltifact

#df = df.sample(frac=0.01)

# Snopes
label2id = {
    'false':2,
    'true':0,
    'mostly true':0,
    'mostly false': 2,
}

rob_large_fever_id_map = { 0:2, 2:0, 1:1}

rob_base_fever_id_map = { 0:2, 1:0, 2:1}

# bart-large-mnli
bart_large_fever_id_map = {    0:2,    1:1,    2:0}

deberta_fever_id_map = {    0:2,    1:0,    2:1}

model_dict = {
    "xlnet" : {"hg_hub_name": "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli", "id_map": None, "short_name": "xlnet"},
    "deberta" : {"hg_hub_name": "cross-encoder/nli-deberta-v3-small", "id_map": deberta_fever_id_map, "short_name": "deberta"},
}
CLAIM_TEXT_VAR_NAME = "claim_text"
EVIDENCE_TEXT_VAR_NAME = "evidence"
LABEL_VAR_NAME = "original_label"

MAX_CLAIM_TOK_LEN = 128
MAX_EVIDENCE_TOK_LEN = 128

def concat_tokenized(evidence_tokens, claim_tokens, model):
    new_tokens = {}
    
    evidence_input_ids = evidence_tokens['input_ids']
    evidence_masks = evidence_tokens['attention_mask']
    
    claim_input_ids = claim_tokens['input_ids']
    claim_masks = claim_tokens['attention_mask']
    
    # Every tokenizer needs to be treated differently
    # XLNet: This is sentence number one <sep> This is the second sentence <sep> <cls> 
    # Input ids: <sep>=4 and <cls>=3
    
    # Roberta/Bart/Deberta: <s> This is sentence number one </s></s> This is the second sentence </s>
    # Input ids: <s>=0 and </s>=2
    
    # remove the last token as it is an ending of seq token.
    # the claim tokens will be concatenated which means 
    # the sequence will not end here.
    if model == "roberta":
        claim_input_ids[:,0] = 2
    elif model == "xlnet":
        evidence_input_ids = evidence_input_ids[:,0:-1]
        evidence_masks = evidence_masks[:,0:-1]
    elif model == "electra":
        claim_input_ids = claim_input_ids[:,1::]
        evidence_masks = evidence_masks[:,1::]
    elif model == "bart":
        claim_input_ids[:,0] = 2
    elif model == "deberta":
        claim_input_ids = claim_input_ids[:,1::]
        evidence_masks = evidence_masks[:,1::]
    else:
        raise Exception("Model not recognized!")

    new_tokens['input_ids'] = torch.cat((evidence_input_ids, claim_input_ids), 1)
    new_tokens['attention_mask'] = torch.cat((evidence_masks, claim_masks), 1)
    
    return new_tokens

# Make configurable dataset that can handle FEVER/FEVEROUS/DeClare/SciFact

class pl_dataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, label2id, model_name, tokenizer, max_claim_token_len: int, max_evidence_token_len: int, claim_var: str, evidence_var: str, label_var: str):
        
        self.label2id = label2id
        
        self.tokenizer = tokenizer
        self.df_data = df
        self.max_claim_token_len = max_claim_token_len
        self.max_evidence_token_len = max_evidence_token_len
        self.label_var_name = label_var
        self.claim_var_name = claim_var
        self.evidence_var_name = evidence_var
        self.model_name = model_name
        
    def classes(self):
        return list(self.label2id.keys())
    
    def label_names(self):
        return list(self.label2id.values())

    def __len__(self):
        return len(self.df_data)

    # Fetch label id.
    def get_label_id(self, data_row: pd.Series):
        return self.label2id[data_row[self.label_var_name]]
    
    # Fetch tokenized claim.
    def get_claim(self, data_row: pd.Series):
        return self.tokenizer(data_row[self.claim_var_name],
                                            padding='max_length',
                                            max_length=self.max_claim_token_len,
                                            return_token_type_ids=True, truncation=True,
                                            return_tensors="pt")

    # Fetch tokenized evidence.
    def get_evidence(self, data_row: pd.Series):
        
        evidence_token_list = []
        
        if data_row[self.evidence_var_name]:
            for evid in data_row[self.evidence_var_name]:
                evidence_token_list.append(self.tokenizer(evid,
                                     padding='max_length',
                                     max_length = self.max_evidence_token_len, 
                                     return_token_type_ids=True,
                                     truncation=True,
                                     return_tensors="pt"))
        else:
            evidence_token_list.append(self.tokenizer("",
                                     padding='max_length',
                                     max_length = self.max_evidence_token_len, 
                                     return_token_type_ids=True,
                                     truncation=True,
                                     return_tensors="pt"))
        
        return evidence_token_list
    

    def __getitem__(self, idx):

        data_row = self.df_data.iloc[idx]
        
        evidences = self.get_evidence(data_row)
        claims = []
        
        
        claims = [self.get_claim(data_row)]*len(evidences)
        
        # Concat the claim tokens to the evidence tokens.
        # Note that the token type ids define which tokens are
        # part of the premise and of the hypothesis in case of 
        # textual entailment
        claims_evidences = [concat_tokenized(evidence, claim, model=self.model_name) for evidence, claim in zip(evidences, claims)]

        #label_id = self.get_label_id(data_row)
        
        return claims_evidences #, label_id


def majority_voting(list_predictions, list_probs):
    
    entailment_scores = []
    neutral_scores = []
    contradiction_scores = []
    
    for pred, prob in zip(list_predictions, list_probs):
        
        if pred == 0:
            entailment_scores.append(prob)
        elif pred == 1:
            neutral_scores.append(prob)
        else:
            contradiction_scores.append(prob)
    
    entailment_score = 0
    neutral_score = 0
    contradiction_score = 0
    
    if len(entailment_scores):
        entailment_score = sum(entailment_scores)/len(entailment_scores)
    
    if len(neutral_scores):
        neutral_score = sum(neutral_scores)/len(neutral_scores)

    if len(contradiction_scores):
        contradiction_score = sum(contradiction_scores)/len(contradiction_scores)
        
    scores = [entailment_score, neutral_score, contradiction_score]
    print(scores)
    
    return scores.index(max(scores)), max(scores), scores

def run_model(model, dataloader, id_map, half=True):
    
    y_pred = []
    y_true = []

    for data in tqdm(dataloader):
        

        #claim_evidences_list, true_label = data
        claim_evidences_list = data
        y_pred_list = []
        y_prob_list = []
        #print(true_label)
        for claim_evidences in claim_evidences_list:

            input_ids = claim_evidences['input_ids'].long().squeeze(1).to(DEVICE)
            attention_mask = claim_evidences['attention_mask'].long().squeeze(1).to(DEVICE)
            with autocast(half):
                outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=None)

            y_pred_id = outputs['logits'].argmax(dim=1).data.cpu().numpy()[0]
            y_prob = torch.max(torch.nn.functional.softmax(outputs['logits'],dim=1)).data.cpu().numpy()
            y_pred_list.append(y_pred_id)
            y_prob_list.append(y_prob)
            
            
        # print(y_pred_list, y_prob_list)
        # break
            
        pred_label, pred_prob, score = majority_voting(y_pred_list, y_prob_list)
        
        normscore = [float(i)/sum(score) for i in score]
        
        if pred_label == 0:
            disinfo_score = 50+(normscore[2]-normscore[0])*100/2
        elif pred_label == 2:
            disinfo_score = 50+(normscore[0]-normscore[2])*100/2
        else:
            disinfo_score = 50+(normscore[2]-normscore[0])*100/2
        
        if id_map:
            pred_label = id_map[pred_label]
        
        # The true label of NOT ENOUGH INFO of FEVER always comes with zero evidence.
        # Predicted the label with zero evidence is therefore arbitrary and always correct.
        #if true_label.data.cpu().numpy()[0] == 1:
        #    pred_label = 1

        y_pred.append(disinfo_score)
        #y_true.append(true_label.data.cpu().numpy()[0])
        
    return y_pred #, y_true
    
def get_entailment(filename, half=False):
    if type(filename) == str:
        data_df = pd.read_csv(filename, 
                delimiter='\t', encoding="UTF-8", index_col=[0], header=0).iloc[:100]
    else:
        data_df = filename
    data_df["content"] = data_df.content.apply(lambda x: x.replace(". . ", ". ") if type(x) != float else "")

    hasContent = ~data_df.isnull().any(axis=1)
    df_test1 = data_df[hasContent]

    
    model_dict_2 = model_dict["xlnet"]

    hg_hub_name = model_dict_2["hg_hub_name"]
    id_map = model_dict_2["id_map"]
    short_name = model_dict_2["short_name"]

    print("Evaluating model "+str(hg_hub_name))

    tokenizer = AutoTokenizer.from_pretrained(hg_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_hub_name)
    model.eval()
    model.to(DEVICE)

    dataset = pl_dataset(df=df_test1, model_name=short_name, label2id=label2id, tokenizer=tokenizer, max_claim_token_len=128, max_evidence_token_len=128, claim_var=claim_var, evidence_var=evid_var, label_var=label_var)
    # dataset = dftest
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    #y_pred, y_true = run_model(model, dataloader, id_map)
    with torch.no_grad():
        y_pred  = run_model(model, dataloader, id_map, half=False)
    
    model.to("cpu")
    df_final = []
    df_final = pd.DataFrame(df_final)
    df_final['claim'] = df_test1['claim']
    df_final['disinfo_score'] = y_pred
    df_final['content'] = df_test1['content']
    
    return df_final


if __name__ == "__main__":

    result = get_entailment("er_processed_sbertqa_sentence_top_3.csv")
    # data_df = pd.read_csv("er_processed_sbertqa_sentence_top_3.csv", 
    #             delimiter='\t', encoding="UTF-8", index_col=[0], header=0).iloc[:100]

    # data_df["content"] = data_df.content.apply(lambda x: x.replace(". . ", ". ") if type(x) != float else "")

    # df_test1 = data_df[~data_df.isnull().any(axis=1)]


    # for key in model_dict.keys():
        
    #     model_dict_2 = model_dict[key]
        
    #     hg_hub_name = model_dict_2["hg_hub_name"]
    #     id_map = model_dict_2["id_map"]
    #     short_name = model_dict_2["short_name"]
        
    #     print("Evaluating model "+str(hg_hub_name))
        
    #     tokenizer = AutoTokenizer.from_pretrained(hg_hub_name)
    #     model = AutoModelForSequenceClassification.from_pretrained(hg_hub_name)
    #     model.eval()
    #     model.to(DEVICE)
            
    #     dataset = pl_dataset(df=df_test1, model_name=short_name, label2id=label2id, tokenizer=tokenizer, max_claim_token_len=128, max_evidence_token_len=128, claim_var=claim_var, evidence_var=evid_var, label_var=label_var)
    #     # dataset = dftest
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
    #     #y_pred, y_true = run_model(model, dataloader, id_map)
    #     with torch.no_grad():
    #         y_pred  = run_model(model, dataloader, id_map, half=False)
        
    #     model.to("cpu")
    #     del model
    #     #print(classification_report(y_true, y_pred, target_names=["true", "not_enough_evidence", "false"]))