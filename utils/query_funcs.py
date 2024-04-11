#List of uncertainty measures: https://towardsdatascience.com/active-learning-overview-strategies-and-uncertainty-measures-521565e0b0b

def random_query(data_loader, query_size = 10):
    indexes = []

    for batch in data_loader:

        indexes.extend(batch["index"].tolist())

        if len(indexes) >= query_size:
            break

    return indexes[0:query_size]

def least_confidece(model, device, data_loader, query_size):
    pass

def margin_of_confidence(model, device, data_loader, query_size):
    pass

def ratio(model, device, data_loader, query_size):
    pass

def entropy(model, device, data_loader, query_size):
    pass