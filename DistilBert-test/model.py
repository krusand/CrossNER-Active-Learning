import torch.nn as nn
from transformers import DistilBertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DistilBertModel
from transformers import DistilBertForTokenClassification
from transformers import AutoModelForTokenClassification
#*****************
#***   Model   ***
#*****************

class BertForTokenClassification(DistilBertForTokenClassification):
    config_class = DistilBertConfig

    def __init__(self, config, tags):
        super().__init__(config)
        self.num_labels = len(tags)
        # Load model body
        self.bert = DistilBertModel(config)
        # Set up token classification head
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Use model body to get encoder representations
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
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
    
#*******************
#***   Dataset   ***
#*******************

