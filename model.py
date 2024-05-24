import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel

import utils.NERutils as nu


#**************************
#***   Early Stopping   ***
#**************************

# Implementation from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation f1 doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation f1 improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation f1 improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_f1, model):

        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\nValidation f1 decreased to {val_f1:.6f}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        '''Saves model when validation f1 increases.'''
        if self.verbose:
            self.trace_func(f'Validation f1 increased ({self.val_f1_max:.6f} --> {val_f1:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1

#*****************
#***   Model   ***
#*****************

class BertForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, tags, patience=3, delta=0, verbose=False, filter_padding = False):
        super().__init__(config)
        self.num_labels = len(tags)
        
        # Load model body
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Load and initialize weights
        self.init_weights()

        # Define patience and delta for early stopping
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.filter_padding = filter_padding

        # Save loss and f1
        self.training_f1 = []
        self.validation_f1 = []

        self.training_loss = []
        self.validation_loss = []

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
            if self.filter_padding:
                mask_filter = attention_mask.type(torch.bool).view(-1)
                loss = loss_fct(logits.view(-1, self.num_labels)[mask_filter], labels.view(-1)[mask_filter])
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return model output object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def train_loop(self, data_loader, device, optimizer):
        self.train()
        
        batch_preds, batch_targets = [], []

        # Initialize parameters for calculating training loss and f1
        num_batches = len(data_loader)
        epoch_loss = 0

        for idx, batch in enumerate(data_loader):
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["labels"].to(device, dtype=torch.long)
            
            outputs = self.forward(input_ids = ids,
                            attention_mask = mask,
                            labels = targets)
            
            loss, logits = outputs.loss, outputs.logits

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Flatten targets and predictions
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # Mask predictions and targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            batch_preds.append(predictions)
            batch_targets.append(targets)

            # Calculate train loss and f1
            epoch_loss += loss.item()

        # Caluclate training loss for the current epoch
        train_loss = epoch_loss/num_batches
        train_f1 = self.get_f1_score(preds=torch.cat(batch_preds, dim=0).cpu().numpy(), targets=torch.cat(batch_targets, dim=0).cpu().numpy())
        
        # Save loss and f1 to history
        self.training_loss.append(train_loss)
        self.training_f1.append(train_f1)

    def val_loop(self, data_loader, device):
        self.eval()
        
        batch_preds, batch_targets = [], []

        # Initialize parameters for calculating training loss
        num_batches = len(data_loader)
        epoch_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                targets = batch["labels"].to(device, dtype=torch.long)
                
                outputs = self.forward(input_ids = ids,
                                attention_mask = mask,
                                labels = targets)

                # Save validation loss
                loss, logits = outputs.loss, outputs.logits

                # Flatten targets and predictions
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                active_logits = logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                
                # Mask predictions and targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                batch_preds.append(predictions)
                batch_targets.append(targets)
                
                # Calculate train loss
                epoch_loss += loss.item()
        
        # Caluclate training loss for the current epoch
        val_loss = epoch_loss/num_batches
        val_f1 = self.get_f1_score(preds=torch.cat(batch_preds, dim=0).cpu().numpy(), targets=torch.cat(batch_targets, dim=0).cpu().numpy())
        
        # Save loss and f1 to history
        self.validation_loss.append(val_loss)
        self.validation_f1.append(val_f1)

    def get_f1_score(self, preds, targets):
    
        _, index2tag, _ = nu.load_vocabs()

        preds = [*map(index2tag.get, list(preds))]
        golds = [*map(index2tag.get, list(targets))]
        f1_score = nu.getF1ScoreFromLists(golds, preds)

        return f1_score

    def fit(self, num_epochs, train_data_loader, val_data_loader, device, optimizer, path):
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, delta=self.delta, path=path)

        for epoch in range(num_epochs):

            if self.verbose:
                print(f"Epoch {epoch+1} of {num_epochs} epochs", flush = True)
            
            print("Train")
            self.train_loop(train_data_loader, device, optimizer)
            print("Validate")
            self.val_loop(val_data_loader, device)
            
            # Early stopping
            early_stopping(self.validation_f1[-1], self)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Done!")


    def predict(self, data_loader, device):
        """Predict logits with a dataloader

        Args:
            data_loader (torch.utils.data.DataLoader): Dataloader for the data to predict
            device (torch.device): Pytorch device used to calculate predictions

        Returns:
            tuple (list,list,list) : Tuple containing the predicted logits, corresponding masks, and indices.

        Returned elements:
            - Logits: The raw outputs of the model before applying any activation function.
            - Masks: Masks indicating which parts of the input data is padding. This can be used, to filter them out in further calculations
            - Indices: Indices from the original dataset.

        """
        self.eval()

        # Initialize parameters for calculating training loss
        logits = []
        masks = []
        indices = []
        # size = len(data_loader.dataset) / data_loader.batch_size
        size = len(data_loader.sampler.indices) / data_loader.batch_size
        size = np.ceil(size).astype("int")

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                
                outputs = self(ids, mask)

                logits.extend(outputs[0])
                masks.extend(mask)
                indices.extend(batch["index"])

        return (logits, masks, indices)
        
