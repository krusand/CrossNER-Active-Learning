# Imports
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import numpy as np
import torch


from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

from seqeval.metrics import f1_score

import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel

# Specify parameters for training
epochs = 0.01
batch_size = 16

MODEL_PATH = "bert_dansk_trained"
LOGGING_PATH = "logs"

TRAIN_PATH = "data/train-00000-of-00001.parquet"
DEV_PATH = 'data/dev-00000-of-00001.parquet'
TEST_PATH = 'data/test-00000-of-00001.parquet'

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")
bert_model_name = "bert-base-multilingual-cased"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# Specify hyperparameters
source_domain = 'Web'

# BERT class
class BertForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = len(tags)
        # Load model body
        self.bert = BertModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # Use model body to get encoder representations
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Encode lines
def encode_line(line):
    ents = line["ents"]
    tokenized = bert_tokenizer(line["text"])
    labels = ["O"]

    word_start = 0
    for word in bert_tokenizer.convert_ids_to_tokens(tokenized["input_ids"]):
        if word in ("[CLS]", "[SEP]"):
            continue
        if word.startswith("##"):
            word = word[2:]

        word_start += line["text"][word_start:].find(word)

        if ents:
            if word_start >= ents[0]["start"] and word_start <= ents[0]["end"]:
                labels.append(ents[0]["label"])
            else:
                labels.append("O")
            if ents[0]["end"] <= word_start + len(word):
                ents = ents[1:]
        else:
            labels.append("O")

        # print(line["text"][word_start: word_start+len(word)])
        word_start += len(word)

    labels = [tag2index[x] for x in labels + ["O"]]
    tokenized["labels"] = labels
    # print(pd.DataFrame([i for i in zip(bert_tokenizer.convert_ids_to_tokens(tokenized["input_ids"]),labels)]))
    return tokenized

# Function to compute performance
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list

def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}

# Read data
train = pd.read_parquet(TRAIN_PATH)
dev = pd.read_parquet(DEV_PATH)
test = pd.read_parquet(TEST_PATH)

# Extract source domain data
train_source = train.loc[train['dagw_domain'] == source_domain]
dev_source = dev.loc[dev['dagw_domain'] == source_domain]
test_source = test.loc[test['dagw_domain'] == source_domain]

# convert to Huggingface dataset
train_dataset_raw = Dataset(pa.Table.from_pandas(train_source))
dev_dataset_raw = Dataset(pa.Table.from_pandas(dev_source))
test_dataset_raw = Dataset(pa.Table.from_pandas(test_source))

# Dict over tags and idx
tags = ["O"] + list(set([x[0]["label"] for x in train_dataset_raw["ents"] if x]))
index2tag = {idx: tag for idx, tag in enumerate(tags)}
tag2index = {tag: idx for idx, tag in enumerate(tags)}

# Encode dataset
train_dataset_pd = pd.DataFrame([encode_line(l) for l in train_dataset_raw])
dev_dataset_pd = pd.DataFrame([encode_line(l) for l in dev_dataset_raw])
test_dataset_pd = pd.DataFrame([encode_line(l) for l in test_dataset_raw])

# Convert to Huggingface dataset
train_dataset = Dataset(pa.Table.from_pandas(train_dataset_pd))
dev_dataset = Dataset(pa.Table.from_pandas(dev_dataset_pd))
test_dataset = Dataset(pa.Table.from_pandas(test_dataset_pd))

# Loading pretrained BERT
bert_config = AutoConfig.from_pretrained(
    bert_model_name, num_labels=len(tags), id2label=index2tag, label2id=tag2index
)

bert_model = BertForTokenClassification.from_pretrained(
    bert_model_name, config=bert_config
).to(device)

logging_steps = len(train_dataset) // batch_size
model_name = f"{bert_model_name}-finetuned_ewt"

# Training arguments
num_epochs = epochs
batch_size = batch_size
logging_steps = len(train_dataset) // batch_size
model_name = f"{bert_model_name}-finetuned-panx-de"
training_args = TrainingArguments(
    output_dir=model_name,
    log_level="error",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_steps=1e6,
    weight_decay=0.01,
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
)

def model_init():
    return BertForTokenClassification.from_pretrained(
        bert_model_name, config=bert_config
    ).to(device)

data_collator = DataCollatorForTokenClassification(bert_tokenizer)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=bert_tokenizer,
)

# Train model
print("training")
trainer.train()

# Save model
trainer.save_model(MODEL_PATH)