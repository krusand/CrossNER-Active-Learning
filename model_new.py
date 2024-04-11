import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel

# Timetracker
from tqdm import tqdm

#*****************
#***   Model   ***
#*****************

class BertForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, tags):
        super().__init__(config)
        self.num_labels = len(tags)
        
        # Load model body
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Load and initialize weights
        self.init_weights()

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
            print(idx)
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
        
        for epoch in range(num_epochs):
            self.train_loop(data_loader, device, optimizer)
            print(f"{epoch+1} of {num_epochs} epochs")

    def test(self, data_loader, device):

        self.val_loop(data_loader, device)


    
#*******************
#***   Dataset   ***
#*******************

