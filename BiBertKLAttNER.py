import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
import torch.nn.functional as F

class KnowledgeAttentionBertForNER(nn.Module):
    def __init__(self, bert, bert_kl, num_labels, num_kl_labels,d_a, r):
        super(KnowledgeAttentionBertForNER, self).__init__()
        
        self.num_labels = num_labels

        self.bert_kl = bert_kl
        self.bert = bert
        self.config = bert_kl.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size + r, num_labels)
        self.linear_first = torch.nn.Linear(num_kl_labels, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        kl_label, _, _ = self.bert_kl(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        bert_out,_ = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_all_encoded_layers=False)
        x = torch.tanh(self.linear_first(kl_label))       
        x = self.linear_second(x)       
        attention = self.softmax(x,1)
        sequence_output = torch.cat((attention, bert_out), 2)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states)
    
    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d,dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
