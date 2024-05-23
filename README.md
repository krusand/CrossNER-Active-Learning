# Domain Adaption with Active Learning
This GitHub repository contains the documentation for our results presented in the report "Domain Adaption with Active Learning" and the code necessary to reproduce the results.

We investigate if AL can help improve the performance with small amounts of data in a domain adaptation setting where BERT has initially been pre-fine-tuned on a source domain and hereafter fine-tuned on a target domain using AL.

## Data
Our data was initially not publicly available, but was requested to be made public. We contacted Kenneth Enevoldsen who kindly made the dataset public. Kenneth Enevoldsen contributed to the publishing of DANSK (NER dataset). Information about the dataset can be found here: https://huggingface.co/datasets/chcaa/dansk-ner.

The tags in the dataset used the tagging standard found in Ontonotes 5.0. This means there are 18 tags. We converted these tags to Beginning-Inside-Outside (BIO) tagging using the script `bio_tag_convert.py`. 

## Model
Our model is build on a pre-trained multilingual BERT model. Specifically we used *bert-base-multilingual-cased*. The model structure can be seen in `model.py`, which also uses functions from `NERutils.py`. It is generally recommended to use a GPU supporting CUDA for all model work like fine-tuning and evaluating.

## Pre-fine-tuning
To pre-fine-tune a model means to fit a model to a *source domain*. Pre-fine-tuning is done using `prefinetune.py`. To run the script, one can specify the *source domain (-f), attention_mask (-am), jointly (-j)* arguments. For our source domain model *News*, we ran the script with the following arguments. 
```sh
python prefinetune.py -f "News" -am True
```
To jointly fit on *News* and a target domain, for example *Legal*, we used the following arguments
```sh
python prefinetune.py -f "Legal" -am True -j True
```
The *-am* argument is used to specify whether the padding should be filtered out when calculating the loss. We found that when this is true, the model fitted the dataset a lot better. Therefore we kept this to be true for all fitting.

The model is by default saved to the folder `fine_tuned/regularized/{filter}_finetuned.pt`. The folder can be changed in the script. The fit-method in the model class automatically saves the model as a .pt file. The default save path is to `checkpoint.pt` in the folder the script is running from.

## Active learning setup
The active learning is performed using the script `ActiveLearning.py`. You have to specify three parameters to run the script *source domain, target domain, query strategy*, and optionally specify *attention mask and run_id*. 

Each time new data is labeled, the model is fitted, and we save the f1-score and the amount of data, that the model was trained on.

To run active learning using the source domain *News*, with a target domain *Conversation*, using a *Vocab* query strategy, we used the following code. 
```sh
python ActiveLearning.py -t "Conversation" -s "News" -q "vocab" -am True -ri 1
```

The active learning saves models to `fine_tuned/active_learning/`, and results to `al_results/`. The results saved in `al_results/` are results showing the development of span-f1 as the percentage of data used increases. 

## F1-scores
`f1_scores.py` can be used to test a model on a domain. In the `f1_scores.py` script, you should specify the models and filters you want to evaluate. Furthermore you should specify the model folder and f1 score save path. To test the pre-fine-tuned news model on conversation, social media, legal and the whole dataset, we used the following `f1_scores.py` setup:

```python
# Load data
train_path = "data/BIOtrain.parquet"
dev_path = "data/BIOdev.parquet"
test_path = "data/BIOtest.parquet"

# specify the model path
model_path = "fine_tuned/regularized/"

# specify where the dictionary should be saved
save_path = 'f1_scores/f1_scores.pkl'

models: list[str] = ["News_finetuned"]
filters: list[str] = ["Conversation", "Social Media", "Legal", None]
```


## Plots
`Results.ipynb` contains the code for the plots presented in the report. 


## Datamap
The `datamap` folder contains the scripts to generate the data mapping. `DataMaps.twb` contains plots of different metrics, that are combined in dashboards. 'datamapScript.py' runs for all domains, and saves .csv files with the performance metrics. `sentenceIdsForTableau.py` just saves the sentences together with their ids, to have make it easier to plot. 
