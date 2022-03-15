run in clutrr/ using

```
$ python3 train.py --train data/clutrr-emnlp/data_db9b8f04/1.2,1.3,1.4_train.csv --test data/clutrr-emnlp/data_db9b8f04/1.5_test.csv data/clutrr-emnlp/data_db9b8f04/1.6_test.csv data/clutrr-emnlp/data_db9b8f04/1.7_test.csv data/clutrr-emnlp/data_db9b8f04/1.8_test.csv data/clutrr-emnlp/data_db9b8f04/1.9_test.csv data/clutrr-emnlp/data_db9b8f04/1.10_test.csv 

$ python3 train.py --train data/clutrr-emnlp/data_089907f8/1.2,1.3_train.csv --test data/clutrr-emnlp/data_089907f8/1.4_test.csv data/clutrr-emnlp/data_089907f8/1.5_test.csv data/clutrr-emnlp/data_089907f8/1.6_test.csv data/clutrr-emnlp/data_089907f8/1.7_test.csv data/clutrr-emnlp/data_089907f8/1.8_test.csv data/clutrr-emnlp/data_089907f8/1.9_test.csv data/clutrr-emnlp/data_089907f8/1.10_test.csv --full_reason True
```

run in graphlog/ using

```
$ python3 train.py --train_world world_17
$ python3 train.py --train_world re_0 --v_T_pos 0.3 
$ python3 train.py --train_world re_1 --v_T_pos 0.3 
$ python3 train.py --train_world re_2 --v_T_pos 0.8 
```

More GraphLog datasets can be downloaded at: https://github.com/facebookresearch/GraphLog 

More CLUTRR datasets can be downloaded at: https://github.com/facebookresearch/clutrr

Please noted that each world is called a "rule" in GraphLog's page. 

environment: 

```
graphlog==1.1.0rc1 
networkx==2.5 
numpy==1.18.1 
torch==1.7.0+cu110 
torch-cluster==1.5.8 
torch-geometric==1.6.3 
torch-scatter==2.0.5 
torch-sparse==0.6.8 
torch-spline-conv==1.2.0 
torchaudio==0.7.1 
torchvision==0.8.1+cu110 
```
