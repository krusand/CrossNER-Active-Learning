# Libraries
import torch
import pickle 

from utils.model import BertForTokenClassification
import utils.NERutils as nu

from transformers import AutoConfig, AutoTokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm

print("Dependencies loaded")

# Define tokenizer
bert_model_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# Load data
train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"
test_path = "data/BIOtest.parquet"


models_and_filters = ['Social Media','News','Web','Conversation','Wiki & Books','Legal','dannet']

f1_scores = {model: dict() for model in models_and_filters}

print("Starting model testing")

for model_type in tqdm(models_and_filters):
    for filter in models_and_filters:
        train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=filter)
        dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=filter,tags=train_dataset.tags, index2tag=train_dataset.index2tag, tag2index=train_dataset.tag2index)
        test_dataset = nu.NERdataset(dataset_path=test_path, tokenizer=bert_tokenizer, filter=filter,tags=train_dataset.tags, index2tag=train_dataset.index2tag, tag2index=train_dataset.tag2index)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Set device
        if torch.backends.mps.is_available():
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        # Config
        bert_model_name = "bert-base-multilingual-cased"
        bert_config = AutoConfig.from_pretrained(
            bert_model_name, 
            num_labels=len(train_dataset.tags), 
            id2label=train_dataset.index2tag, 
            label2id=train_dataset.tag2index
        )

        # initialise model
        model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=train_dataset.tags, verbose=True).to(device)

        # Load model
        model.load_state_dict(torch.load(f"/home/aksv/NLP_assignments/Project/fine_tuned/source_domains/{model_type}_finetuned.pt", map_location=device))

        # Evaluate model
        preds, targets = nu.evaluate_model(model=model, dataloader=train_loader, device=device)

        # Convert ids to tags
        preds = [*map(train_dataset.index2tag.get, list(preds))]
        golds = [*map(train_dataset.index2tag.get, list(targets))]

        f1score = nu.getF1ScoreFromLists(golds=golds, preds=preds)
        print(f"{f1score = }\n{model_type = }\n{filter = }\n")
        f1_scores[model_type][filter] = f1score

with open('f1scores_training_data_testing.pkl', 'wb') as fp:
    pickle.dump(f1_scores, fp)


