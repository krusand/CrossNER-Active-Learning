# Import
import pandas as pd
import numpy as np

from transformers import AutoTokenizer



#****************************
#***   Dataset features   ***
#****************************

# Get dataset
def get_dataset(path):
    """
    Input
        path: path to file directory
    Output
        df: pandas dataframe
    """
    
    df = pd.read_parquet(path)

    return df

# Get vocab features
def get_vocab_features(df):
    """
    Input
        df: Pandas dataframe
    
    Output
        tags:
        index2tag:
        tag2index: 
    """

    tags = ["O"] + list(set([x[0]["label"] for x in df["ents"] if len(x)>0]))
    index2tag = {idx: tag for idx, tag in enumerate(tags)}
    tag2index = {tag: idx for idx, tag in enumerate(tags)}

    return tags, index2tag, tag2index

#**************************
#***   Encode dataset   ***
#**************************

# Encode line
def encode_line(line, max_length, tag2index):
    """
    Encode 

    """

    bert_model_name = "bert-base-multilingual-cased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    ents = line["ents"]
    tokenized = bert_tokenizer(line["text"], padding='max_length', truncation = True, max_length=max_length)
    labels = ["O"]

    word_start = 0
    for word in bert_tokenizer.convert_ids_to_tokens(tokenized["input_ids"]):
        if word in ("[CLS]", "[SEP]"):
            continue
        if word.startswith("##"):
            word = word[2:]

        word_start += line["text"][word_start:].find(word)

        if ents.size > 0:
            if word_start >= ents[0]["start"] and word_start <= ents[0]["end"]:
                labels.append(ents[0]["label"])
            else:
                labels.append("O")
            if ents[0]["end"] <= word_start + len(word):
                ents = ents[1:]
        else:
            labels.append("O")

        word_start += len(word)

    labels = [tag2index[x] for x in labels + ["O"]]
    tokenized["labels"] = labels
    
    return tokenized


def find_max_length(df):
    
    bert_model_name = "bert-base-multilingual-cased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    return max([len(bert_tokenizer(line["text"])["input_ids"]) for line in df.iloc])

def encode_lines(df, tag2index):
    
    max_length = find_max_length(df)
    df_encoded = pd.DataFrame([encode_line(l, max_length, tag2index) for l in df.iloc])

    return df_encoded