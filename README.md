# BERT enhanced by medical QA pairs and knowledge labels
Improving BERT Performance on Medical Named Entity Recognition by Online Question-Answering Pairs and Knowledge Labels
### Knowledge Base
[kb_data_all.json]kb_data_all.json

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
