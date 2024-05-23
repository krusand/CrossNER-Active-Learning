
import time
start_time = time.time()

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoConfig, AutoTokenizer

# Model
from model import BertForTokenClassification

# Self made functions
import utils.NERutils as nu
import utils.QueryUtils as q

from tqdm import tqdm
import argparse

### EDIT ###
# Specify the path to the source domain models
path = "fine_tuned/regularized/"

parser = argparse.ArgumentParser(description="NER Active Learning Script")
parser.add_argument("-t", "--target", type=str, help="Specify the target domain for active learning")
parser.add_argument("-s", "--source", type=str, help="Specify the source domain model used for active learning")
parser.add_argument("-q", "--query", type=str, help="Specify the query strategy to be used for active learning")
parser.add_argument("-am", "--attention_mask", type=bool, required= False, help="Specify if the padding is filtered for the loss")
parser.add_argument("-ri", "--run_id", type=int, required= False, help="Specify the run id for the model.")

args = parser.parse_args()

target_domain = args.target
source_domain = args.source
query_strategy = args.query
filter_padding = args.attention_mask if args.attention_mask is not None else True
run_id = args.run_id if args.run_id is not None else ''

print(f"Starting script:\n{query_strategy = }\n{target_domain = }\n{source_domain = }\n{run_id = }\n", flush=True)

num_queries = 20
memory = int(round(int(torch.cuda.get_device_properties(0).total_memory) / 1000000000))
batch_size = memory
max_attempts = memory //2
n_attempts = 0

# Model parameters to specify
num_epochs = 10000
learning_rate = 1e-05
patience = 10

tags, index2tag, tag2index = nu.load_vocabs()

print(torch.cuda.get_device_name(0), flush=True)


###########
def perform_active_learning(batch_size,
                            num_queries,
                            query_strategy, 
                            num_epochs, 
                            learning_rate, 
                            patience,
                            target_domain,
                            source_domain,
                            path,
                            run_id):
    
    model_save_path = "fine_tuned/active_learning/" + f"model_{source_domain}_{target_domain}_{query_strategy}"
    
    # Specify path for data
    train_path = path + "data/BIOtrain.parquet"
    dev_path = path + "data/BIOdev.parquet"

    # Define tokenizer
    bert_model_name = "bert-base-multilingual-cased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    bert_model_name = "bert-base-multilingual-cased"
    bert_config = AutoConfig.from_pretrained(
        bert_model_name, 
        num_labels=len(tags), 
        id2label=index2tag, 
        label2id=tag2index
    )

    # Get data from target domain
    print("Loading data", flush=True)
    train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=target_domain,tags=tags,tag2index=tag2index, index2tag=index2tag)
    dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=target_domain,tags=tags,tag2index=tag2index, index2tag=index2tag)

    # Define dataloader for validation
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False ,num_workers=0)

    # Get source vocab for vocab queries
    if query_strategy == "vocab":
        source_vocab = q.get_source_vocab(source_domain=source_domain, train_path= train_path, bert_tokenizer=bert_tokenizer)
        target_scores = q.get_target_scores(target_domain, train_path, source_vocab, bert_tokenizer)
    else:
        source_vocab = None
        target_scores = None

    # Initialize parameters
    loss = []
    f1_scores = []
    n_samples = []
    p_samples = []
    max_f1 = 0
    
    dataset_size = len(train_dataset)
    pool_size = int(len(train_dataset)*0.2)

    # Initialize with training on random subset
    # Reset model and optimizer 
    print("Recompile model")
    model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=tags, patience=patience, verbose=True, filter_padding = filter_padding).to(device)
    model.load_state_dict(torch.load(path + source_domain + "_finetuned.pt", map_location=device))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay = 0.01)

    # Ask the oracle to label samples using one of the strategies
    print("Query the oracle")
    query_size = int(0.05 * dataset_size)
    q.query_the_oracle(model, device, train_dataset, query_size, query_strategy, pool_size = pool_size, source_vocab=source_vocab, target_scores = target_scores, bert_tokenizer=bert_tokenizer)

    # Create a dataloader with labeled indexes
    labeled_idx = np.where(train_dataset.unlabeled_mask == 0)[0]
    labeled_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(labeled_idx))
    print(f"Number of labeled indexes: {len(labeled_idx)}" , flush = True)

    # train model
    print("Fit model", flush = True)
    model.fit(num_epochs, labeled_loader, dev_loader, device, optimizer, f"{model_save_path}_checkpoint_{run_id}.pt")

    # Find validation loss for history
    val_loss = model.validation_loss[-11]
    val_f1 = model.validation_f1[-11]

    # Calcualte num_samples
    num_samples = len(labeled_idx)
    per_samples = num_samples/dataset_size
    
    # Save model if it outperformed previous model
    if val_f1 > max_f1:
        max_f1 = val_f1
        torch.save(model.state_dict(), model_save_path + f"_{run_id}.pt")
        print("Model saved", flush = True)

    print("Save test values", flush = True)
    print(f"The result of the test was:\nF1 score: {val_f1} with num_samples {num_samples} and per_samples {per_samples}", flush=True)
    loss.append(val_loss)
    f1_scores.append(val_f1)
    n_samples.append(num_samples)
    p_samples.append(per_samples)

    for query in tqdm(range(2, num_queries+1)):
        
        # Reset model and optimizer 
        print("Recompile model")
        model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=tags, patience=patience, verbose=True, filter_padding = filter_padding).to(device)
        model.load_state_dict(torch.load(path + "fine_tuned/" + source_domain + "_finetuned.pt", map_location=device))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay = 0.01)

        # Ask the oracle to label samples using one of the strategies
        print("Query the oracle")
        query_size = int((query*0.05-per_samples) * dataset_size)
        q.query_the_oracle(model, device, train_dataset, query_size, query_strategy, pool_size = pool_size, source_vocab=source_vocab, target_scores = target_scores, bert_tokenizer=bert_tokenizer)

        # Create a dataloader with labeled indexes
        labeled_idx = np.where(train_dataset.unlabeled_mask == 0)[0]
        labeled_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(labeled_idx))
        print(f"Number of labeled indexes: {len(labeled_idx)}" , flush = True)

        # train model
        print("Fit model", flush = True)
        model.fit(num_epochs, labeled_loader, dev_loader, device, optimizer, f"{model_save_path}_checkpoint_{run_id}.pt")

        # Find validation loss for history
        val_loss = model.validation_loss[-11]
        val_f1 = model.validation_f1[-11]

        # Calcualte num_samples
        num_samples = len(labeled_idx)
        per_samples = len(labeled_idx)/len(train_dataset)

        # Save model if it outperformed previous model
        if val_f1 > max_f1:
            max_f1 = val_f1
            torch.save(model.state_dict(), model_save_path + f"_{run_id}.pt")
            print("Model saved", flush = True)

        print("Save test values", flush = True)
        print(f"The result of the test was:\nF1 score: {val_f1} with num_samples {num_samples} and per_samples {per_samples}", flush=True)
        loss.append(val_loss)
        f1_scores.append(val_f1)
        n_samples.append(num_samples)
        p_samples.append(per_samples)

    ALResult = pd.DataFrame({"Loss":loss, "f1": f1_scores, "number_of_samples": n_samples, "percentage_of_samples": p_samples})
    ALResult.to_csv(f"{path}al_results/ALResult_{source_domain}_{target_domain}_{query_strategy}_{run_id}.csv", index = False)
    print(ALResult, flush=True)

while n_attempts < max_attempts and batch_size > 0:

    try:
        perform_active_learning(batch_size=batch_size,
                                num_queries=num_queries,
                                query_strategy=query_strategy,
                                num_epochs=num_epochs,
                                learning_rate=learning_rate,
                                patience=patience,
                                target_domain=target_domain,
                                source_domain = source_domain,
                                path=path,
                                run_id=run_id)

        break
    except torch.cuda.OutOfMemoryError: 
        print(f"MemoryError: Reducing batchsize to batch size {batch_size - 2}", flush = True)
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
