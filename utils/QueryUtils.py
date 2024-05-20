#List of uncertainty measures: https://towardsdatascience.com/active-learning-overview-strategies-and-uncertainty-measures-521565e0b0b

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
import random
from scipy.sparse import csr_array

import utils.NERutils as nu

import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress the SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def query_the_oracle(model, device, dataset, query_size=10, query_strategy='random', 
                     interactive=True, pool_size=0, batch_size=16, num_workers=0, source_vocab = None, target_scores = None, bert_tokenizer = None):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]

    # Pool based sampeling
    if pool_size > 0:
        pool_idx = random.sample(list(unlabeled_idx), min(pool_size, len(unlabeled_idx)))
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
    else:
        pool_idx = None
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=SubsetRandomSampler(unlabeled_idx))
    
    # Strategies
    if query_strategy == 'margin':
        sample_idx = margin_of_confidence(model, device, pool_loader, query_size)

    elif query_strategy == 'confidence':
        sample_idx = least_confidece(model, device, pool_loader, query_size)
    
    elif query_strategy == 'vocab':
        sample_idx = vocab_overlap(target_scores, pool_idx, query_size)
        
    else:
        sample_idx = random_query(pool_loader, query_size)
    
    # Move observation to the pool of labeled samples
    for sample in sample_idx:
        dataset.unlabeled_mask[sample] = 0

def get_vocab(input_ids, df_indices, bert_tokenizer):
    ids = []
    values = []
    row_indices = []
    for index, sentence in zip(df_indices, input_ids):
        i, v = np.unique(sentence, return_counts=True)
        ids.extend(i)
        values.extend(v)
        row_indices.extend(np.repeat(index, i.shape[0]))

    shape = (bert_tokenizer.vocab_size, np.max(row_indices) +1)
    vocab = csr_array((values, (ids, row_indices)), shape=shape)


    for i in [0,100,101,102,103]:
        vocab[i] = 0

    vocab = vocab / vocab.sum()
    
    return vocab

def get_source_vocab(source_domain, train_path, bert_tokenizer):
    train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=source_domain)
    train_params = {'batch_size': 8,
                'num_workers': 0,
                }

    data_loader = torch.utils.data.DataLoader(train_dataset, **train_params)
    return get_vocab(data_loader.dataset.encodings["input_ids"],data_loader.dataset.encodings.index * 0, bert_tokenizer)

def get_target_scores(target_domain, train_path, source_vocab, bert_tokenizer):
    train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=target_domain)
    train_params = {'batch_size': 8,
                'num_workers': 0,
                }

    data_loader = torch.utils.data.DataLoader(train_dataset, **train_params)

    vocab = get_vocab(data_loader.dataset.encodings["input_ids"],data_loader.dataset.encodings.index, bert_tokenizer)
    summed_simmilarities = (vocab*source_vocab).sum(axis=0)
    return np.argsort(summed_simmilarities)
    

def random_query(data_loader, query_size = 10):
    indexes = []

    for batch in data_loader:

        indexes.extend(batch["index"].tolist())

        if len(indexes) >= query_size:
            break

    return indexes[0:query_size]

def least_confidece(model, device, data_loader, query_size = 10):
    """Returns indices for the data with the lowest confidence

    Args:
        model (model_new.BertForTokenClassification): Model used for prediction
        device (torch.device): Pytorch device
        data_loader (torch.utils.data.dataloader.DataLoader): DataLoader with random sampler
        query_size (int, optional): Number of elements to return. Defaults to 10.

    Returns:
        np.array: Array of indeces
    """
    logits, mask, indices = model.predict(data_loader, device)
    probabilities = [F.softmax(r, dim = 1) for r in logits]

    confidences = [torch.max(r, dim=1).values for r in probabilities] 
    summed_confidences = [torch.multiply(c, m.float()).cpu().max() for c,m in zip(confidences, mask)]

    sorted_idx = np.argsort(summed_confidences)
    return np.asarray(indices)[sorted_idx][0:query_size]

def margin_of_confidence(model, device, data_loader, query_size):
    """Returns indices for the data with the lowest margin of confidence

    Args:
        model (model_new.BertForTokenClassification): Model used for prediction
        device (torch.device): Pytorch device
        data_loader (torch.utils.data.dataloader.DataLoader): DataLoader with random sampler
        query_size (int, optional): Number of elements to return. Defaults to 10.

    Returns:
        np.array: Array of indeces
    """
    logits, mask, indices = model.predict(data_loader, device)

    probabilities = [F.softmax(sentence, dim = 1) for sentence in logits]

    toptwo = [torch.topk(sentence, 2, dim=1)[0] for sentence in probabilities]
    difference = [torch.abs(sentence[:,0]-sentence[:,1]) for sentence in toptwo]

    summed_margins = [torch.multiply(c, m.float()).cpu().max() for c,m in zip(difference, mask)]

    sorted_idx = np.argsort(summed_margins)

    return np.asarray(indices)[sorted_idx][0:query_size]

def vocab_overlap(scores, pool_idx, query_size):
    if pool_idx:
        scores = np.intersect1d(scores, pool_idx)
    return scores[0:query_size]