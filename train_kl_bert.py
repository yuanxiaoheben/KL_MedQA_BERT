import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
from utils import *
from evaluate import *
import pandas as pd
import os
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from BiBertKLAttNER import KnowledgeAttentionBertForNER
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch.nn import CrossEntropyLoss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")

parser.add_argument("--validation_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input validation corpus.")

parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        required=False,
                        help="model learning rate.")

parser.add_argument("--epochs",
                        default=20,
                        type=int,
                        required=False,
                        help="model training epochs.")

parser.add_argument("--weight_decay",
                        default=7e-4,
                        type=float,
                        required=False,
                        help="Weight Decay.")

parser.add_argument("--kl_path",
                        default=None,
                        type=str,
                        required=True,
                        help="knowledge label model path.")

parser.add_argument("--bert_path",
                        default=None,
                        type=str,
                        required=True,
                        help="pre-trained BERT model path.")

parser.add_argument("--batch_size",
                        default=15,
                        type=int,
                        required=False,
                        help="batch size")

parser.add_argument("--max_seq_length",
                        default='90',
                        type=int,
                        required=False,
                        help="maximum sequence length")

parser.add_argument("--saved_path",
                        default=None,
                        type=str,
                        required=True,
                        help="save path.")
args = parser.parse_args()



ner_val_data = pd.read_csv(args.validation_corpus)
ner_train_data = pd.read_csv(args.train_corpus)


train_ner_set = ner_list(ner_train_data)
val_ner_set = ner_list(ner_val_data)

tag_to_ix = {"O": 0, "B-D": 1, "B-T": 2,"B-S": 3,"B-C": 4,"B-P": 5,"B-B": 6,
             "D": 7, "T": 8,"S": 9,"C": 10,"P": 11,"B": 12
             }
max_seq_length = args.max_seq_length
folder_path = args.saved_path

pad_token_label_id = CrossEntropyLoss().ignore_index

def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=pad_token_label_id,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = tag_to_ix

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        for word, label in zip(example[0], example[1]):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append([input_ids,input_mask,segment_ids,label_ids])
    return features

def load_and_cache_examples(input_data, max_seq_length, pad_token_label_id=pad_token_label_id):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    features = convert_examples_to_features(input_data, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to("cuda")
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long).to("cuda")
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long).to("cuda")
    all_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long).to("cuda")

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def train_ner(model, optimizer, epochs, train_dataset, test_dataset):
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    for j in range(epochs):
        loss_sum = 0
        for i,batch in enumerate(train_loader):
            model.train()
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            loss_sum = loss_sum+loss.sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            model.zero_grad()
        
        print( "Epochs %i; Loss %f" %(j+1, loss_sum))
        torch.save(model, os.path.join(folder_path, 'ner.epo.' + str(j+1) + '.model'))

def test_ner(model, test_dataset):
    model.eval()
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    target_list = []
    out_list = []
    for i,batch in enumerate(test_loader):
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            outputs = model(**inputs)
            preds = outputs[0].detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()
            preds = np.argmax(preds, axis=2)
            target_list += [x for x in out_label_ids]
            out_list += [x for x in preds]
    return target_list, out_list

train_dataset = load_and_cache_examples(train_ner_set, max_seq_length, pad_token_label_id)
val_dataset = load_and_cache_examples(val_ner_set, max_seq_length, pad_token_label_id)
num_kl_labels =10
hyper_parameters = {
    'num_kl_labels':10,
    'd_a':100,
    'r':100,
}
print(hyper_parameters)
d_a = 100
r = 100
num_kl_labels = 10
kl_model = torch.load(args.kl_path)
bert_model = BertModel.from_pretrained(args.bert_path)
ner_model = KnowledgeAttentionBertForNER(bert_model, kl_model, len(tag_to_ix), hyper_parameters['num_kl_labels'],
    hyper_parameters['d_a'], hyper_parameters['r'])
ner_model.to("cuda")
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
weight_decay = args.weight_decay
# warmup_linear = WarmupLinearSchedule(warmup=0.1)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in ner_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay},
    {"params": [p for n, p in ner_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate)
train_ner(ner_model, optimizer, epochs, train_dataset, val_dataset)