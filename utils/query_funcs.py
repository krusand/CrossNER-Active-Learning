#List of uncertainty measures: https://towardsdatascience.com/active-learning-overview-strategies-and-uncertainty-measures-521565e0b0b

import torch.nn.functional as F
import torch
import numpy as np

def random_query(data_loader, query_size = 10):
    indexes = []

    for batch in data_loader:

        indexes.extend(batch["index"].tolist())

        if len(indexes) >= query_size:
            break

    return indexes[0:query_size]

def least_confidece(model, device, data_loader, query_size = 10):
    logits, mask, indices = model.predict(data_loader, device)
    probabilities = [F.softmax(r, dim = 1) for r in logits]
    confidences = [torch.max(r, dim=1).values for r in probabilities] 
    summed_confidences = [torch.dot(c, m.float()).cpu() for c,m in zip(confidences, mask)]
    sorted_idx = np.argsort(summed_confidences)
    return np.asarray(indices)[sorted_idx][0:query_size]

def margin_of_confidence(model, device, data_loader, query_size):
    pass

def ratio(model, device, data_loader, query_size):
    pass

def entropy(model, device, data_loader, query_size):
    pass