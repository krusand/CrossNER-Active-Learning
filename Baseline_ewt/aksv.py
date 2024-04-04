import pandas as pd
from transformers import AutoTokenizer
import torch

bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def read_iob2_with_metadata(file_path):
    documents = []
    with open(file_path, 'r') as f:
        document = []
        metadata = {}
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if line.startswith('# text'):
                    metadata['text'] = line.split('=')[1].strip()  
                continue
            if not line:  
                if document: 
                    documents.append((metadata, document))
                    document = []  
                    metadata = {}  
            else:
                parts = line.split('\t')
                if len(parts) >= 4:
                    token = parts[1]
                    ner_tag = parts[2]
                    document.append((token, ner_tag))
        if document:  
            documents.append((metadata, document))
    return documents


training_data = read_iob2_with_metadata('/home/aksv/NLP_assignments/Project/data/ewt_data/en_ewt-ud-dev.iob2')
dev_data = read_iob2_with_metadata('/home/aksv/NLP_assignments/Project/data/ewt_data/en_ewt-ud-train.iob2')



def convert_to_df(data):
    sentences = []
    tags = []

    for line in data:
        sentences.append(line[0]["text"])
        prev_tag = ''
        for i,word_tag in enumerate(line[1]):
            _, tag = word_tag
            if i == 0:
                prev_tag += tag
            else:
                prev_tag += "," + tag

        tags.append(prev_tag)


    return pd.DataFrame({"sentence": sentences, "tags": tags})

training_df = convert_to_df(training_data)
dev_df = convert_to_df(dev_data)


label2id = {k: v for v, k in enumerate(pd.unique(','.join(training_df["tags"].values).split(',')))}
id2label = {v: k for v, k in enumerate(pd.unique(','.join(training_df["tags"].values).split(',')))}

MAX_LEN = 512
def check_if_sentence_is_longer_than_max_len_and_truncate_if(sentence: list, padding):
    if len(sentence) > MAX_LEN:
        sentence = sentence[:MAX_LEN]
    else:
        sentence = sentence + [padding for _ in range(MAX_LEN - len(sentence))]
    return sentence

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = ["[CLS]"]
    labels = ["O"]
    
    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    tokenized_sentence.append("[SEP]")
    labels.append("O")

    tokenized_sentence = check_if_sentence_is_longer_than_max_len_and_truncate_if(tokenized_sentence, "[PAD]")
    labels = check_if_sentence_is_longer_than_max_len_and_truncate_if(labels, "O")


    return tokenized_sentence, labels

def get_attn_mask(tokenized_sentence):
    return [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]


def get_proper_data_structure(df):
    df["tokenized_sentence"] = df.apply(lambda x: tokenize_and_preserve_labels(x["sentence"],x["tags"], tokenizer)[0],axis=1)
    df["tokenized_sentence_tags"] = df.apply(lambda x: tokenize_and_preserve_labels(x["sentence"],x["tags"], tokenizer)[1],axis=1)

    df["ids"] = df.apply(lambda x: tokenizer.convert_tokens_to_ids(x["tokenized_sentence"]), axis=1)
    df["attn_mask"] = df.apply(lambda x: get_attn_mask(x["tokenized_sentence"]), axis=1)
    df["targets"] = df.apply(lambda x: [label2id[label] for label in x["tokenized_sentence_tags"]],axis=1)
    return df

training_df = get_proper_data_structure(training_df)
dev_df = get_proper_data_structure(dev_df)

def get_torch_dataset(df):
    data_set = []

    for ids, attn, targets in df[["ids","attn_mask", "targets"]].iloc:
        data_set.append({"ids":torch.tensor(ids),
                            "mask":torch.tensor(attn),
                            "targets":torch.tensor(targets)})
    return data_set

training_set = get_torch_dataset(training_df)
dev_set = get_torch_dataset(dev_df)

train_params = {'batch_size': 32,
                'shuffle': True,
                'num_workers': 1
                }


training_loader = torch.utils.data.DataLoader(training_set, **train_params)


from transformers import BertForTokenClassification

device = torch.device("cuda")

model = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)
model.to(device)

LEARNING_RATE = 1e-05
N_EPOCHS = 5
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

from tqdm import tqdm
def run_epoch(epoch):
    model.train()

    tr_loss, tr_accuracy = 0, 0

    for idx, batch in enumerate(tqdm(training_loader)):
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["targets"].to(device, dtype=torch.long)
        
        outputs = model(input_ids = ids,
                        attention_mask = mask,
                        labels = targets)
        
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(tr_loss)


for epoch in range(N_EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    run_epoch(epoch)



