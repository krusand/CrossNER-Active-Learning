import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import os

import pandas as pd
import numpy as np
import random

from utils.model_new import BertForTokenClassification
import utils.NERutils as nu

from transformers import AutoConfig, AutoTokenizer

print("Imports loaded")

N_EPOCHS = 100
LEARNING_RATE = 1e-05

# Define tokenizer
bert_model_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"
test_path = "data/BIOtest.parquet"

filter = 'Legal'

train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=filter)
dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer)
test_dataset = nu.NERdataset(dataset_path=test_path, tokenizer=bert_tokenizer, filter=filter)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Config
bert_model_name = "bert-base-multilingual-cased"
bert_config = AutoConfig.from_pretrained(
    bert_model_name, 
    num_labels=len(train_dataset.tags), 
    id2label=train_dataset.index2tag, 
    label2id=train_dataset.tag2index
)

model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=train_dataset.tags, verbose=True).to(device)
print("Loaded Model")

def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[beg][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans



def getF1ScoreFromLists(golds:list, preds: list):
    tp = 0
    fp = 0
    fn = 0
    for goldEnt, predEnt in zip(golds, preds):
        goldSpans = toSpans(goldEnt)
        predSpans = toSpans(predEnt)
        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap
        
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1


print("Beginning training")

model.fit(num_epochs=N_EPOCHS, 
          train_loader=train_loader, 
          dev_loader=dev_loader,
          device=device, 
          optimizer=torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=0.01))






