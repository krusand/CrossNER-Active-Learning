
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
path = "/home/aksv/NLP_assignments/Project/fine_tuned/source_domains_reg/"

# baseline_model_paths = ["Legal_finetuned", "Conversation_finetuned", "dannet_finetuned", "News_finetuned", "Social Media_finetuned", "Web_finetuned", "Wiki & Books_finetuned"]

# baseline_model_paths = ["Legal_finetuned"]
# baseline_model_paths = ["Conversation_finetuned"]
# baseline_model_paths = ["dannet_finetuned"]
# baseline_model_paths = ["News_finetuned"]
baseline_model_paths = ["Social Media_finetuned"]
# baseline_model_paths = ["Web_finetuned"]
# baseline_model_paths = ["Wiki & Books_finetuned"]


target_domains = ['Legal','Web', 'Conversation','News','Wiki & Books','Social Media', 'dannet']

# target_domains = ['Legal']
# target_domains = ['Web']
# target_domains = ['Conversation']
# target_domains = ['News']
# target_domains = ['Wiki & Books']
# target_domains = ['Social Media']
# target_domains = ['dannet']


query_strategies = ["random", "margin", "confidence"]

# query_strategies = ["random"]
# query_strategies = ["margin"]
# query_strategies = ["confidence"]


batch_size = 75
query_size = 100
pool_size = 500

# Model parameters to specify
num_epochs = 10000
learning_rate = 1e-05
patience = 10

print(torch.cuda.get_device_name(0))

###########
def perform_active_learning(batch_size, 
                            query_size, 
                            pool_size, 
                            query_strategies, 
                            num_epochs, 
                            learning_rate, 
                            patience,
                            target_domains,
                            baseline_model_paths,
                            path):

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

    for baseline_model_path in baseline_model_paths:
        for target_domain in target_domains:
            for query_strategy in query_strategies:
                # Get data from target domain
                train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=target_domain)
                dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=target_domain,tags=train_dataset.tags, index2tag=train_dataset.index2tag, tag2index=train_dataset.tag2index)

                # Define dataloader for validation
                dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False ,num_workers=0)

                print(query_strategy, target_domain, baseline_model_path)
                # Initialize parameters
                loss = []
                acc = []
                f1_scores = []
                n_samples = []
                p_samples = []
                min_loss, min_acc = np.inf, np.inf
                
                model_path = "/home/aksv/NLP_assignments/Project/fine_tuned/active_learning_models/multiple_gpu/" + f"model_{baseline_model_path}_{target_domain}_{query_strategy}.pt"
                num_queries = len(train_dataset)//query_size

                for query in range(num_queries):
                    
                    # Reset model and optimizer 
                    print("Recompile model")
                    model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=train_dataset.tags, patience=patience, verbose=True).to(device)
                    model.load_state_dict(torch.load(path + baseline_model_path + ".pt", map_location=device))
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
                    preds, targets = nu.evaluate_model(model, dev_loader, device)
                    preds = [*map(train_dataset.index2tag.get, list(preds))]
                    golds = [*map(train_dataset.index2tag.get, list(targets))]
                    f1 = nu.getF1ScoreFromLists(golds, preds)
                    
                    print("Save test values")
                    loss.append(val_loss)
                    acc.append(val_acc)
                    f1_scores.append(f1)
                    n_samples.append(num_samples)
                    p_samples.append(per_samples)

                    # Save model if it outperformed previous model
                    print("Save model")
                    if val_loss < min_loss:
                        min_loss = val_loss
                        torch.save(model.state_dict(), model_path)
                        print("Model saved")

                    ALResult = pd.DataFrame({"Loss":loss, "Accuracy": acc, "f1": f1_scores, "number_of_samples": n_samples, "percentage_of_samples": p_samples})
                    ALResult.to_csv(f"/home/aksv/NLP_assignments/Project/al_results/multiple_gpu/ALResult_{baseline_model_path}_{target_domain}_{query_strategy}.csv", index = False)


max_attempts = 75
n_attempts = 1



while n_attempts < max_attempts and batch_size > 0:

    try:
        perform_active_learning(batch_size=batch_size,
                                query_size=query_size,
                                pool_size=pool_size,
                                query_strategies=query_strategies,
                                num_epochs=num_epochs,
                                learning_rate=learning_rate,
                                patience=patience,
                                target_domains=target_domains,
                                baseline_model_paths=baseline_model_paths,
                                path=path)

        break
    except Exception as e: 
        print(e)
        print(f"Reducing batchsize to batch size {batch_size - 5}")
        batch_size -= 5
        n_attempts += 1
        torch.cuda.empty_cache()
else:
    print("Max attempts used")


