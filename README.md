# Moral Choice Machine
Code repository for the Moral Choice Machine: ...

## 1. Structure
The code is structured in:

* /data: pre-processing of the datasets
* /mcm: the Moral Choice Machine bias calculation
* /sentenceEmbedding: retraining of sentence Embeddings with the Universal Sentence Encoder

The retraining of the USE is based on the code: https://github.com/tensorflow/models/tree/master/research/skip_thoughts
## 2. Dependencies
    tensorflow>=1.12.0
    tensorflow_hub>=0.3.0
    gensim>=3.8.1
    nltk>=3.4.5
    numpy>=1.17.2
    scipy>=1.3.1
    tqdm>=4.36.1
    pandas>=0.25.1
    
## 3. Using the MCM

To run the MCM based on the pre-trained USE emeddings (tensorflow hub):

~~~~
python experiments/experiments_mcm.py --data <atomic/context> --model use_hub
~~~~

## 4. Retraining the underlying embedding

~~~~
TODO

python experiments/experiments_mcm.py --data <atomic/context> --model train_rc
~~~~

## 5. Citing & Authors

If you find this repository helpful, feel free to cite our publication:
TODO