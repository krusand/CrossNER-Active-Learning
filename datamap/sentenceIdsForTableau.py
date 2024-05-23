import pandas as pd
train_path = "data/BIOtrain.parquet"
df = pd.read_parquet(path=train_path)

all_domains = []
for d in set(df["dagw_domain"]):
    l = list(df["text"][df["dagw_domain"] == d])
    domain = pd.DataFrame(l, index = range(len(l)), columns=["sentence"])
    domain["domain"] = d
    all_domains.append(domain)

domain = pd.concat(all_domains)
domain.to_csv("sentences.csv")


