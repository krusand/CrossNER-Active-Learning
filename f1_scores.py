# Libraries
import time
start_time = time.time()

import torch
import pickle

from model import BertForTokenClassification
import utils.NERutils as nu

from transformers import AutoConfig, AutoTokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm

print("Dependencies loaded", flush=True)

device_name = torch.cuda.get_device_name(0)
memory = int(round(int(torch.cuda.get_device_properties(0).total_memory) / 1000000000))

print(device_name, memory, flush=True)

# Define tokenizer
bert_model_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
tags, index2tag, tag2index = nu.load_vocabs()

# Load data
train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"
test_path = "data/BIOtest.parquet"
al_model_path = "/home/aksv/NLP_assignments/Project/fine_tuned/active_learning/third/"
ft_model_path = "/home/aksv/NLP_assignments/Project/fine_tuned/regularized/third/"
# models = ["news_and_Conversation_finetuned","news_and_Social Media_finetuned", "news_and_Legal_finetuned"]


models = [
"model_News_Social Media_margin",
"model_News_Social Media_confidence",
"model_News_Social Media_random",
"model_News_Legal_margin",
"model_News_Legal_confidence",
"model_News_Legal_random",
"model_News_Conversation_margin",
"model_News_Conversation_confidence",
"model_News_Conversation_random",
"Conversation_finetuned",
"News_finetuned",
"Legal_finetuned",
"Social Media_finetuned",
"news_and_Conversation_finetuned",
"news_and_Legal_finetuned",
"news_and_Social Media_finetuned"]

model_paths = [al_model_path,
al_model_path,
al_model_path,
al_model_path,
al_model_path,
al_model_path,
al_model_path,
al_model_path,
al_model_path,
ft_model_path,
ft_model_path,
ft_model_path,
ft_model_path,
ft_model_path,
ft_model_path,
ft_model_path]

filters = [None]

batch_size = memory

f1_scores = {model: dict() for model in models}

print("Starting model testing", flush=True)

for model_type,model_path in zip(models, model_paths):
    print(f"Starting model {model_type}", flush=True)

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
        num_labels=len(tags), 
        id2label=index2tag, 
        label2id=tag2index
    )

    # initialise model
    model = BertForTokenClassification.from_pretrained(bert_model_name, config=bert_config, tags=tags, verbose=True).to(device)

    # Load model
    model.load_state_dict(torch.load(f"{model_path}{model_type}.pt", map_location=device))

    model.eval()

    for filter in filters:
        print(f"Starting filter {filter}",flush=True)
        train_dataset = nu.NERdataset(dataset_path=train_path, tokenizer=bert_tokenizer, filter=filter, tags=tags,tag2index=tag2index, index2tag=index2tag)
        dev_dataset = nu.NERdataset(dataset_path=dev_path, tokenizer=bert_tokenizer, filter=filter, tags=tags,tag2index=tag2index, index2tag=index2tag)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Evaluate model
        preds, targets = nu.evaluate_model(model=model, dataloader=dev_loader, device=device)

        # Convert ids to tags
        preds = [*map(index2tag.get, list(preds))]
        golds = [*map(index2tag.get, list(targets))]

        f1score = nu.getF1ScoreFromLists(golds=golds, preds=preds)
        print(f"{f1score = }\n{model_type = }\n{filter = }\n")
        f1_scores[model_type][filter] = f1score

    save_path = 'f1_scores/f1_scores_whole_dataset.pkl'

    print(save_path)

    with open(save_path, 'wb') as fp:
        pickle.dump(f1_scores, fp)

end_time = time.time()
run_time = (end_time - start_time)

hours = int(run_time // 3600)
minutes = int((run_time % 3600) // 60)
seconds = int(run_time % 60)

print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds")
