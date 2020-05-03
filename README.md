# BERT enhanced by medical QA pairs and knowledge labels
Improving BERT Performance on Medical Named Entity Recognition by Online Question-Answering Pairs and Knowledge Labels
### Knowledge Base
Knowledge base is in: [kb_data_all.json](https://github.com/yuanxiaoheben/KL_MedQA_BERT/blob/master/kb_data_all.json). In the file, the key is a word's name in Chinese, value is the word's knowledge classes, and it represent by "1" and "0". "1" means it appear in a class, and "0" means it disappear in a class. The classes order is: people, medical instrument, operation, diagnosis, disease, syndrome, medicine, sport, position, and food.

For example: "blood tension": "0001000000" means "blood tension" in the diagnosis class.

### Train
Using [bert_ensemble_train.py](https://github.com/yuanxiaoheben/DeepDuSite/blob/master/bert_pytorch/bert_ensemble_train.py) for training.
 ``` shell
 python bert_ensemble_train.py \
 --train_corpus train_label.csv \
 --validation_corpus valid_label.csv \
 --bert_path bert.model.ep9 \
 --saved_path ./
 ```
 
 ### Test
 Using [bert_ensemble_test.py](https://github.com/yuanxiaoheben/DeepDuSite/blob/master/bert_pytorch/bert_ensemble_test.py) for test.
 ``` shell
 python bert_ensemble_test.py \
 --test_corpus test_label.csv \
 --model_path bert.ensemble.ep1 
 ```
