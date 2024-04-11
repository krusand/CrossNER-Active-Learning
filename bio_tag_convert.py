import pandas as pd

def convert_ents_to_bio_tagging(row: pd.core.series.Series) -> dict:
    def create_dict(start: int, end:int, ent:str) -> dict:
        d = {"start": start,
             "end": end,
             "label":ent}
        
        return d
    
    bio_list = []

    tokens, ents, text = row["tokens"], row["ents"], row["text"]
    
    for token in tokens:
        bio_dict = {}
        token_start, token_end = token["start"], token["end"]
        token_txt = text[token_start:token_end]

        for ent in ents:
            ent_start = ent["start"]
            ent_end = ent["end"]
            ent_label = ent["label"]

            if token_start == ent_start:
                bio_list.append(create_dict(token_start, token_end, 'B-' + ent_label))
            elif token_start > ent_start and token_end <= ent_end:
                bio_list.append(create_dict(token_start, token_end, 'I-' + ent_label))
    return bio_list

def convert_df(df):
    df["ents"] = df.apply(convert_ents_to_bio_tagging, axis=1)
    return df

convert_df(pd.read_parquet("data/train.parquet")).to_parquet("data/BIOtrain.parquet")
convert_df(pd.read_parquet("data/dev.parquet")).to_parquet("data/BIOdev.parquet")
convert_df(pd.read_parquet("data/test.parquet")).to_parquet("data/BIOtest.parquet")