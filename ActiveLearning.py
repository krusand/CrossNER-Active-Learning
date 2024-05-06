
# Basic
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

### EDIT ###
# Parameters to specify
baseline_model_path = "" ### EDIT
target_domains = ['News', 'Web', 'Conversation', 'Social Media', 'Wiki & Books', 'Legal']

batch_size = 32
query_size = 20
pool_size = 200
query_strategies = ["random", "margin", "confidence"]

# Model parameters to specify
num_epochs = 100
learning_rate = 1e-05
patience = 3

###########

# Specify path for data
train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"
test_path = "data/BIOtest.parquet"

# Define tokenizer
bert_model_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# Pretrained Bert Model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Config
train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer)

bert_model_name = "bert-base-multilingual-cased"
bert_config = AutoConfig.from_pretrained(
    bert_model_name, 
    num_labels=len(train_dataset.tags), 
    id2label=train_dataset.index2tag, 
    label2id=train_dataset.tag2index
)


# Initialize parameters
loss = []
acc = []
f1_scores = []
n_samples = []
p_samples = []
min_loss, min_acc = np.inf, np.inf

for target_domain in target_domains:

    # Get data from target domain
    train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=target_domain)
    dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=target_domain)
    test_dataset = nu.NERdataset(dataset_path=test_path, tokenizer=bert_tokenizer, filter=target_domain)

    # Define dataloader for validation
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    for query_strategy in query_strategies:
        model_path = f"model_{target_domain}_{query_strategy}.pt"
        num_queries = len(train_dataset)//query_size

        for query in range(num_queries):
            
            # Reset model and optimizer 
            print("Recompile model")
            model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=train_dataset.tags, patience=patience, verbose=True).to(device)
            model.load_state_dict(torch.load(baseline_model_path, map_location=device))
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

            # Ask the oracle to label samples using one of the strategies
            print("Query the oracle")
            q.query_the_oracle(model, device, train_dataset, query_size, query_strategy, pool_size)

            # Create a dataloader with labeled indexes
            labeled_idx = np.where(train_dataset.unlabeled_mask == 0)[0]
            labeled_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(labeled_idx))
            print(f"Number of labeled indexes: {len(labeled_idx)}")

            # train model
            print("Fit model")
            model.fit(num_epochs, labeled_loader, dev_loader, device, optimizer, model_path)

            # Find validation loss and accuracy for history
            val_loss = model.validation_loss[-1]
            val_acc = model.validation_acc[-1]

            # Calcualte num_samples
            num_samples = len(labeled_idx)
            per_samples = len(labeled_idx)/len(train_dataset)

            # Calculate f1
            preds, targets = nu.evaluate_model(model, test_loader, device)
            preds = [*map(train_dataset.index2tag.get, list(preds))]
            golds = [*map(train_dataset.index2tag.get, list(targets))]
            f1 = nu.getF1ScoreFromLists(golds, preds)
            
            print("Save test values")
            loss.append(val_loss)
            acc.append(val_acc)
            f1_scores.append(f1)
            n_samples.append(num_samples)
            p_samples.append(per_samples)

            # Save model if it outperform previous model
            print("Save model")
            if val_loss < min_loss:
                torch.save(model.state_dict(), model_path)

            print("Model saved")

        ALResult = pd.DataFrame({"Loss":loss, "Accuracy": acc, "f1": f1_scores, "number_of_samples": n_samples, "percentage_of_samples": p_samples})
        ALResult.to_csv(f"ALResult_{target_domain}_{query_strategy}.csv")



