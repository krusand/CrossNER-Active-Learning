# Import
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

#****************************
#***   Dataset features   ***
#****************************

# Get dataset
def readDataset(path: str) -> pd.DataFrame:
    """
    Reads dataset from a path

    :param path: path to file. Must be .parquet format

    :returns: a pandas dataframe containing the data stored in the specified file
    """
    
    df = pd.read_parquet(path=path)

    return df

# Get vocab features
def getVocabFeatures(df: pd.DataFrame) -> tuple[list, dict, dict]:
    """
    Gets NER labels, and dictionaries mapping indexes to tags, and tags to indexes, from the specified dataframe 

    :param df: Pandas dataframe
    
    :returns: a list containing NER labels, a dictionary mapping index to tag, a dictionary mapping tag to index
    """

    tags = set()
    for row in df.iloc:
        ents = row["ents"]
        if len(ents) == 0:
            continue
        temp_tag = set()
        for ent in ents:
            temp_tag.add(ent["label"])
        tags.update(temp_tag)

    tags = ["O"] + list(tags)

    index2tag = {idx: tag for idx, tag in enumerate(tags)}
    tag2index = {tag: idx for idx, tag in enumerate(tags)}

    return tags, index2tag, tag2index


""" class NERdataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer: AutoTokenizer) -> None:
        self.__df = readDataset(dataset_path)
        self.tags, self.index2tag, self.tag2index = getVocabFeatures(self.__df)
        self.MAX_LENGTH = max(findMaxLength(self.__df, tokenizer), 512)
        self.encodings = encodeDataFrame(self.__df, tokenizer, self.tag2index, self.MAX_LENGTH)

    def __len__(self) -> int:
        return len(self.__df)
    
    def __getitem__(self, index: int) -> dict[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        return item """


class NERdataset(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 tokenizer: AutoTokenizer, 
                 unlabeled_frac: float = None, 
                 filter: str = None,
                 tags = None,
                 index2tag = None,
                 tag2index = None) -> None:
        self.__df = readDataset(dataset_path)
        if tags is None and index2tag is None and tag2index is None:
            self.tags, self.index2tag, self.tag2index = getVocabFeatures(self.__df)
        else:
            self.tags, self.index2tag, self.tag2index = tags, index2tag, tag2index
        
        if filter is not None:
            self.__df = self.__df[self.__df['dagw_domain']==filter]
        self.MAX_LENGTH = max(findMaxLength(self.__df, tokenizer), 512)
        self.encodings = encodeDataFrame(self.__df, tokenizer, self.tag2index, self.MAX_LENGTH)

        # Set all as unlabeled
        self.unlabeled_mask = np.ones(len(self.__df))

    def __len__(self) -> int:
        return len(self.__df)
    
    def __getitem__(self, index: int) -> dict[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item.update({"index": index})
        return item


#************************
#******  ML utils  ******
#************************

def findMaxLength(df: pd.DataFrame, tokenizer: AutoTokenizer) -> int:
    """
    Finds maximum length of text in a given dataframe. Assumes dataframe has column "text". Tokenizes text, and finds the number of tokens.

    :param df: Pandas DataFrame
    :param tokenizer: Pretrained AutoTokenizer

    :returns: maximum token length

    """
    return max([len(tokenizer(line["text"])["input_ids"]) for line in df.iloc])
   

def encodeDataFrame(df: pd.DataFrame, tokenizer: AutoTokenizer, tag2index: dict, max_length: int = 512) -> pd.DataFrame:
    """
    
    :param df: Pandas DataFrame
    :param tokenizer: Pretrained AutoTokenizer 
    :param tag2index: Dictionary mapping tags to indexes
    :param max_length: Max token length
    :param filter: Filter to filter on one domain

    :returns: Pandas DataFrame with colums ["attention_mask", "input_ids", "labels", "token_type_ids"]
    """
    
    def areThereEntitiesLeft(ents: list) -> bool:
        return ents.size > 0

    def isEndOfEntityReached(end_idx: int, token: str, token_start: int) -> bool:
        return end_idx <= (token_start + len(token))

    encoded_list = []

    for row in df.iloc:
            
        ents = row["ents"]
        text = row["text"]
        labels = ["O"]
        token_start = 0

        tokenized_text = tokenizer(text, padding='max_length', truncation = True, max_length=max_length)
        
        for token in tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"]):
            if token in ("[CLS]", "[SEP]"):
                # skip to next token
                continue 
            if token.startswith("##"):
                token = token[2:]

            token_start += row["text"][token_start:].find(token)
            if areThereEntitiesLeft(ents):
                start_idx = ents[0]["start"]
                end_idx = ents[0]["end"]
                entity_label = ents[0]["label"]

                if start_idx <= token_start <= end_idx:
                    labels.append(entity_label)
                else:
                    labels.append("O")
                if isEndOfEntityReached(end_idx, token, token_start):
                    # remove that entity
                    ents = ents[1:]
            else:
                labels.append("O")

            token_start += len(token)

        labels = [tag2index[label] for label in labels + ["O"]]
        tokenized_text["labels"] = labels
        encoded_list.append(tokenized_text)
    
    return pd.DataFrame(encoded_list)


#***************************
#*****   Predictions   *****
#***************************

def readPredictions(path: str, sep: str) -> list[list[str]]:
    """
    Converts predictions to a list of lists containing entity tags

    :param path: path to file containing predictions
    :param sep: separator used in the file containing the predictions 

    :returns: a list of lists, where each list contains tags
    """
    
    with open(path, 'r', encoding = 'utf-8') as file:
        entsList = [line.strip().split(sep) for line in file]
        file.close()
    
    return entsList


def getEntsForPredictions(df: pd.DataFrame) -> list[list[str]]:
    """
    Returns entities for predictions

    :param df: dataframe containing columns ["ents", "tokens"]

    :returns: a list of lists, where each list contains tags
    """
    
    def entsToList(line):
        ents = line['ents']
        tokens = line['tokens']
        ents_list = []

        for token in tokens:
            start = token['start']
            end = token['end']

            if len(ents) > 0:
                ent = ents[0]
                ent_start = ent['start']
                ent_end = ent['end']
                ent_label = ent['label']

                if end == ent_end:
                    ents_list.append(ent_label)
                    ents = ents[1:]
                elif start >= ent_start and end < ent_end:
                    ents_list.append(ent_label)
                else:
                    ents_list.append('O')
                    
            else:
                ents_list.append('O')
        
        return ents_list
    
    ent_list = []
    for row in df.iloc:
        ent_list.append(entsToList(row))
    return ent_list



#***************************
#*****   Evaluation   *****
#***************************

def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans


def getF1ScoreFromPath(predPath: str, goldPath: str):
    gold = readDataset(goldPath)
    pred =  readDataset(predPath)
    goldEnts = getEntsForPredictions(gold)
    predEnts =  getEntsForPredictions(pred)
    entScores = []
    tp = 0
    fp = 0
    fn = 0
    for goldEnt, predEnt in zip(goldEnts, predEnts):
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


def evaluate_model(model, dataloader, device):
        
    batch_preds, batch_targets = [], []

    model.eval()
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["labels"].to(device, dtype=torch.long)
            
            outputs = model.forward(input_ids = ids,
                            attention_mask = mask,
                            labels = targets)
            
            # Save validation loss
            loss, tr_logits = outputs.loss, outputs.logits

            # Flatten targets and predictions
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # Mask predictions and targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            batch_preds.append(predictions)
            batch_targets.append(targets)

    return torch.cat(batch_preds, dim=0).cpu().numpy(), torch.cat(batch_targets, dim=0).cpu().numpy()




def main() -> None:
    return None

if __name__ == '__main__':
    main()

