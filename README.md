# Transformer-Encoder generated context-aware embeddings for  spell correction

Official implementation

To train the model:
```
python train --input input_file.csv --eval eval_file.csv --epochs 100 --batch_size 128
```
The default arguments were optimized for NVIDIA RTC 1080 Ti GPU. Additional configuration details can be found in `config.py`.

`Dockerfile` can be used to replicate the running environment.



### Abstract
In this paper, we propose a novel approach for context-aware spell correction in text documents. We present a deep learning model that learns a context-aware character-level mapping of words to a compact embedding space. In the embedding space, a word and its spell variations are mapped close to each other in Euclidean distance. After we develop this mapping for all words in the dataset’s vocabulary, it is possible identify and correct wrongly spelt words by comparing the distances of their mappings with those of the correctly spelt words. In this space, Euclidean distance can be deemed as a context-aware string similarity metric. Further, the embeddings also capture context of the word, which enables us to identify contextual misspellings like their/there, your/you’re, piece/peace etc and correct them.


We employ a transformer-encoder model that takes character-level input of words and their context to achieve this. The embeddings are generated as output of the model. The model is then trained to minimize triplet loss, which ensures that spell variants of a word are embedded close to the word, and that unrelated words are embedded farther away. Since our model also captures context, words that have similar spellings when spelt correctly but appear in different contexts (e.g., piece/peace) would not be close-by in the embedding space. We further improve the efficiency of training by using a hard triplet mining approach. 


## Authors
1. Noufal **kvsnoufal@gmail.com**
2. Dr. Hani Ragab **h.ragabhassen@hw.ac.uk**

