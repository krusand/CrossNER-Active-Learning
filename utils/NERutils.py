# Import
import pandas as pd
import numpy as np

from transformers import AutoTokenizer


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

    tags = ["O"] + list(set([x[0]["label"] for x in df["ents"] if len(x)>0]))
    index2tag = {idx: tag for idx, tag in enumerate(tags)}
    tag2index = {tag: idx for idx, tag in enumerate(tags)}

    return tags, index2tag, tag2index

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
   

def encodeDataFrame(df: pd.DataFrame, tokenizer: AutoTokenizer, tag2index: dict, max_length: int = 500, filter: str = None) -> pd.DataFrame:
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
    if filter is not None:
        df = df[filter]
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




def main() -> None:
    return None

if __name__ == '__main__':
    main()
