#### version 0.2
Train on real data
- filtering the real samples by deleting sentences and shorten long trajects (max 60)
- shuf and split dataset into train.txt and test.txt, set test.txt size to 64xn
- python tool/create_dataset.py
- set the embedding in models/crnn.py to 30
- set max_iter in val to 100
- set --valInterval to 1000
- set --saveInterval to 1000
- python crnn_main.py --adadelta --cuda

#### version 0.1
Support var length training by 
- Create LMDB Dataset: sort training list according to traject length with random shift
- Load LMDB Dataset: use randomSequentialSampler to load continues batch and padding a batch list of trajects to the same length

Usage:
1. prepare train.txt, test.txt
2. run tool/create_dataset.py
3. python crnn_main.py --adadelta --cuda


#### version 0.0
modified from https://github.com/meijieru/crnn.pytorch
