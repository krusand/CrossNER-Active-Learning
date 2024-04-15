import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel

# Timetracker
from tqdm import tqdm


#**************************
#***   Early Stopping   ***
#**************************

# Implementation from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\nValidation loss increased to {val_loss:.6f}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#*****************
#***   Model   ***
#*****************

class BertForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, tags, patience=3, delta=0, verbose=False):
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

        # Save accuracy and loss
        self.training_acc = []
        self.training_loss = []

        self.validation_acc = []
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

        # Initialize parameters for calculating training loss and accuracy
        num_batches = len(data_loader)
        size = len(data_loader.dataset)
        epoch_loss, correct = 0, 0

        for idx, batch in enumerate(tqdm(data_loader)):
            print(f"Number of batches: {num_batches}")
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["labels"].to(device, dtype=torch.long)
            
            outputs = self.forward(input_ids = ids,
                            attention_mask = mask,
                            labels = targets)
            
            loss, tr_logits = outputs.loss, outputs.logits

            # Flatten targets and predictions
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # Mask predictions and targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate train loss and accuracy
            epoch_loss += loss.item()
            correct += (targets == predictions).type(torch.float).sum().item()
        
        # Caluclate training loss and accuracy for the current epoch
        train_loss = epoch_loss/num_batches
        train_acc = correct/size
        
        # Save loss and accuracy to history
        self.training_loss.append(train_loss)
        self.training_acc.append(train_acc)

    def val_loop(self, data_loader, device):
        self.eval()

        # Initialize parameters for calculating training loss and accuracy
        num_batches = len(data_loader)
        size = len(data_loader.dataset)
        epoch_loss, correct = 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                targets = batch["labels"].to(device, dtype=torch.long)
                
                outputs = self.forward(input_ids = ids,
                                attention_mask = mask,
                                labels = targets)
                
                # Save validation loss
                loss, tr_logits = outputs.loss, outputs.logits

                # Flatten targets and predictions
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                active_logits = tr_logits.view(-1, self.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                
                # Mask predictions and targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                # Calculate train loss and accuracy
                epoch_loss += loss.item()
                correct += (targets == predictions).type(torch.float).sum().item()
        
        # Caluclate training loss and accuracy for the current epoch
        val_loss = epoch_loss/num_batches
        val_acc = correct/size
        
        # Save loss and accuracy to history
        self.validation_loss.append(val_loss)
        self.validation_acc.append(val_acc)

    def fit(self, num_epochs, data_loader, device, optimizer):
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, delta=self.delta)

        for epoch in range(num_epochs):

            if self.verbose:
                print(f"Epoch {epoch+1} of {num_epochs} epochs")
           
            self.train_loop(data_loader, device, optimizer)
            self.val_loop(data_loader, device)
            
            # Early stopping
            early_stopping(self.validation_loss[-1], self)

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

        # Initialize parameters for calculating training loss and accuracy
        logits = []
        masks = []
        indices = []
        # size = len(data_loader.dataset) / data_loader.batch_size
        size = len(data_loader.sampler.indices) / data_loader.batch_size
        size = np.ceil(size).astype("int")

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader), total=size):
                
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                # targets = batch["labels"].to(device, dtype=torch.long)
                
                outputs = self(ids, mask)
                
                # Save validation loss
                # print(type(outputs[0]))
                # outputs = [o[torch.nonzero(m)] for o,m in zip(outputs[0], mask)]
                # print(outputs[0].shape, mask.shape)
                # print(torch.nonzero(mask))
                logits.extend(outputs[0])
                masks.extend(mask)
                indices.extend(batch["index"])

        return (logits, masks, indices)
        


    def test(self, data_loader, device):

        self.val_loop(data_loader, device)
    
#*******************
#***   Dataset   ***
#*******************

