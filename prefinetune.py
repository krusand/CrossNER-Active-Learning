# Libraries
import time
start_time = time.time()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import os
import pickle
import pandas as pd
import numpy as np
import random

from model import BertForTokenClassification
import utils.NERutils as nu
import argparse

from transformers import AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser(description="NER Fine-tuning Script")
parser.add_argument("-f", "--filter", type=str, required=False, help="Specify the filter for NER training")
parser.add_argument("-am", "--attention_mask", type=bool, required= False, help="Specify if the padding is filtered for the loss")
args = parser.parse_args()

filter_padding = args.attention_mask
filter = ["News", args.filter] 
print(filter, flush=True)


device_name = torch.cuda.get_device_name(0)
memory = int(round(int(torch.cuda.get_device_properties(0).total_memory) / 1000000000))
print(device_name, memory, flush=True)

print("Imports loaded", flush=True)

N_EPOCHS = 10000
LEARNING_RATE = 1e-05

batch_size = memory
n_attempts = 0
max_attempts = batch_size

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Define tokenizer
bert_model_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"

tags, index2tag, tag2index = nu.load_vocabs()

# filters = ['Social Media','News','Web','Conversation','Wiki & Books','Legal','dannet']



while n_attempts < max_attempts and batch_size > 0:
    try:
        train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=filter, tags=tags,tag2index=tag2index, index2tag=index2tag)
        dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=filter, tags=tags,tag2index=tag2index, index2tag=index2tag)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Config
        bert_model_name = "bert-base-multilingual-cased"
        bert_config = AutoConfig.from_pretrained(
            bert_model_name, 
            num_labels=len(tags), 
            id2label=index2tag, 
            label2id=tag2index
        )

        model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, patience=25, tags=tags, verbose=True, filter_padding=filter_padding).to(device)
        print("Loaded Model", flush=True)

        print("Beginning training", flush=True)

        model.fit(num_epochs=N_EPOCHS, 
                train_data_loader=train_loader, 
                val_data_loader=dev_loader,
                device=device, 
                optimizer=torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=0.01),
                path=f"fine_tuned/regularized/third/news_and_{args.filter}_2_finetuned.pt")

        print(f"Model: 'third/{filter}_2_finetuned' finished")
        break
    except Exception as e: 
        print(f"MemoryError. Reducing batchsize to batch size {batch_size - 2}", flush=True)
        batch_size -= 2
        n_attempts += 1
        torch.cuda.empty_cache()
else:
    print("Max attempts used")


end_time = time.time()
run_time = (end_time - start_time)

hours = int(run_time // 3600)
minutes = int((run_time % 3600) // 60)
seconds = int(run_time % 60)

print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds")

