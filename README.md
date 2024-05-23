# CrossNER-Active-Learning  
This GitHub repository contains the documentation for our results presented in the report "Domain Adaption with Active Learning" and the code necessary to reproduce the results.

We investigate if AL can help improve the performance with small amounts of data in a domain adaptation setting where BERT has initially been pre-fine-tuned on a source domain and hereafter fine-tuned on a target domain using AL.

### Data
Our data is not public available and but was requested by Kenneth Enevoldsen, who contributed to the publishing of a Danish NER dataset. Information about the datset can be found here: https://huggingface.co/datasets/chcaa/dansk-ner.

The tags in the dataset has been converted to BIO-format using the file bio_tag_convert.py.

### Model
Our model is build on a pre-trained multilingual BERT model. The model structure can be seen in model.py, which also draws from NERutils.

### Pre-fine-tuning
prefinetune.py can be used to perform the pre-fine-tuning step. The model is saved during the training. You can specify a filter if you want the model to train on a specific domain. The fit-method in the model class automatically saves the model as a pt-file. You can specify a path, if you want it to be saved in a specific name. Oterwise it will be saved as "Checkpoint.pt".

### Active learning setup
The active learning is performed using the script "ActiveLearning.py". You have to specify five parameters to run the script: source domain, target domain, query strategy, attention mask and run_id. Attention mask is used to specify wether the padding should be filtered out when calculating the loss. We set it to true during all training.

Each time new data is labeled, the model is fitted, and we save the f1-score and the amount of data, that the model was trained on.

### F1-scores
"f1-scores.py" can be used to test a specific model on a specific domain. You have to specify the path to the model and the domain, you want the model to be tested on. If you set filter = [None], the model will be tested on all domains.

### Plots
"Results.ipynb" contains the code for the plots presented in the report.