import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class BertSequenceMultiLabel(nn.Module):
    def __init__(self, bert, num_labels):
        super(BertSequenceMultiLabel, self).__init__()
        
        self.num_labels = num_labels

        self.bert = bert
        self.config = bert.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_all_encoded_layers=False)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1, self.num_labels)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
