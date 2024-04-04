import pandas as pd
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import pyarrow as pa
from transformers import AutoConfig
from transformers import BertForTokenClassification
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import Trainer

EPOCHS = 1
MODEL_PATH = "bert_trained"
LOGGING_PATH = "logs"

TRAIN_PATH = "data/ewt_data/en_ewt-ud-train.iob2"
DEV_PATH = 'data/ewt_data/en_ewt-ud-dev.iob2'

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
bert_model_name = "bert-base-multilingual-cased"


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


    return tokenized_sentence, labels

def get_attn_mask(tokenized_sentence):
    return [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]


def get_proper_data_structure(df):
    df["tokenized_sentence"] = df.apply(lambda x: tokenize_and_preserve_labels(x["sentence"],x["tags"], tokenizer)[0],axis=1)
    df["tokenized_sentence_tags"] = df.apply(lambda x: tokenize_and_preserve_labels(x["sentence"],x["tags"], tokenizer)[1],axis=1)

    df["input_ids"] = df.apply(lambda x: tokenizer.convert_tokens_to_ids(x["tokenized_sentence"]), axis=1)
    df["attention_mask"] = df.apply(lambda x: get_attn_mask(x["tokenized_sentence"]), axis=1)
    df["labels"] = df.apply(lambda x: [label2id[label] for label in x["tokenized_sentence_tags"]],axis=1)
    return df


training_data = read_iob2_with_metadata(TRAIN_PATH)
dev_data = read_iob2_with_metadata(DEV_PATH)

training_df = convert_to_df(training_data)
dev_df = convert_to_df(dev_data)

label2id = {k: v for v, k in enumerate(pd.unique(','.join(training_df["tags"].values).split(',')))}
id2label = {v: k for v, k in enumerate(pd.unique(','.join(training_df["tags"].values).split(',')))}

training_df = get_proper_data_structure(training_df)
dev_df = get_proper_data_structure(dev_df)


training_set = pd.DataFrame(training_df[["input_ids", "attention_mask", "labels"]])
dev_set = pd.DataFrame(dev_df[["input_ids", "attention_mask", "labels"]])

training_set_pa = Dataset(pa.Table.from_pandas(training_set))
dev_set_pa = Dataset(pa.Table.from_pandas(dev_set))



bert_config = AutoConfig.from_pretrained(bert_model_name, 
                                         num_labels=len(id2label),
                                         id2label=id2label, label2id=label2id)
def model_init():
    return (BertForTokenClassification
            .from_pretrained(bert_model_name, config=bert_config)
            .to(device))

logging_steps = len(training_set) // batch_size
model_name = f"{bert_model_name}-finetuned_ewt"

training_args = TrainingArguments(
    output_dir=model_name, log_level="error", num_train_epochs=EPOCHS, 
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size, evaluation_strategy="epoch", 
    save_steps=1e6, weight_decay=0.01, disable_tqdm=False, 
    logging_steps=logging_steps, push_to_hub=False, logging_dir=LOGGING_PATH)

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)


def data_collator_new(data):
    data = data_collator(data)
    for key in data:
        data[key] = data[key].to(device)
    return data


trainer = Trainer(model_init=model_init, args=training_args, 
                  data_collator = data_collator_new,
                  train_dataset = training_set_pa,
                  eval_dataset = dev_set_pa,
                  tokenizer = tokenizer)


trainer.train()

trainer.save_model(MODEL_PATH)




