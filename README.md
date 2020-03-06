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


## 3. Using the MCM interactive

To run the MCM based on the pre-trained USE embeddings (tensorflow hub):

~~~~
python experiments/experiments_mcm.py --data <atomic/context> --model use_hub
~~~~
    
## 4. Reproducing the results

To run the MCM based on the pre-trained USE embeddings (tensorflow hub):

~~~~
python experiments/mcm_answering.py --model use_hub
~~~~

## 5. Retraining the underlying embedding

~~~~
1. Prepare the dataset
    python data/preprocessBooks.py --data_dir <source directory> --out_dir <destination directory>
2. Preprocess the data for the retraining
    python sentenceEmbedding/skip_thoughts/data/preprocess_dataset_use.py --input_dir <directory/to/dataset/folder/> --output_dir <directory/to/output/> --train_output_shards <number of shards> --num_validation_sentences <# of validation sentences> 
3. Run the retraining
    CUDA_VISIBLE_DEVICES=# python sentenceEmbedding/skip_thought/train_use.py --input_file_pattern <'/path/to/preprocessed/data/train-?????-of-12345'> --input_file_pattern_snli <'/path/to/preprocessed/snli/data/train-?????-of-12345'> --train_dir <output directory> --learning_weight_decoder <weight>

Using the retrained model:
python experiments/experiments_mcm.py --data <atomic/context> --model train_rc
~~~~

## 6. Citing & Authors

If you find this repository helpful, feel free to cite our publication:
TODO