# BERT enhanced by medical QA pairs and knowledge labels
Improving BERT Performance on Medical Named Entity Recognition by Online Question-Answering Pairs and Knowledge Labels
### Knowledge Base
Knowledge base is in: [kb_data_all.json](https://github.com/yuanxiaoheben/KL_MedQA_BERT/blob/master/kb_data_all.json). In the file, the key is a word's name in Chinese, value is the word's knowledge classes, and it represent by "1" and "0". "1" means it appear in a class, and "0" means it disappear in a class. The classes order is: people, medical instrument, operation, diagnosis, disease, syndrome, medicine, sport, position, and food.

For example: "blood tension": "0001000000" means "blood tension" in the diagnosis class.

### Train
Using [train_kl_bert.py](https://github.com/yuanxiaoheben/DeepDuSite/blob/master/bert_pytorch/bert_ensemble_train.py) for training.
 ``` shell
 python train_kl_bert.py \
--train_corpus ../weak_label/train_ner_0929.csv \
--validation_corpus ../weak_label/test_ner_0926.csv \
--bert_path ../tcm_bert/finetuned_lm_domain_corpus/pytorch_bert_model.11.tar.gz \
--kl_path ../weak_label/saved/bert.kl.1.model \
--saved_path ./
 ```
 
