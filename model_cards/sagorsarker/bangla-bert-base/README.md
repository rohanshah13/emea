---
language: bn
tags:
- bert
- bengali
- bengali-lm
- bangla
license: MIT
datasets:
- common_crawl
- wikipedia
- oscar
---


# Bangla BERT Base
A long way passed. Here is our **Bangla-Bert**! It is now available in huggingface model hub. 

[Bangla-Bert-Base](https://github.com/sagorbrur/bangla-bert) is a pretrained language model of Bengali language using mask language modeling described in [BERT](https://arxiv.org/abs/1810.04805) and it's github [repository](https://github.com/google-research/bert)



## Pretrain Corpus Details
Corpus was downloaded from two main sources:

* Bengali commoncrawl copurs downloaded from [OSCAR](https://oscar-corpus.com/)
* [Bengali Wikipedia Dump Dataset](https://dumps.wikimedia.org/bnwiki/latest/)

After downloading these corpus, we preprocessed it as a Bert format. which is one sentence per line and an extra newline for new documents. 

```
sentence 1
sentence 2

sentence 1
sentence 2

```

## Building Vocab
We used [BNLP](https://github.com/sagorbrur/bnlp) package for training bengali sentencepiece model with vocab size 102025. We preprocess the output vocab file as Bert format.
Our final vocab file availabe at [https://github.com/sagorbrur/bangla-bert](https://github.com/sagorbrur/bangla-bert) and also at [huggingface](https://huggingface.co/sagorsarker/bangla-bert-base) model hub.

## Training Details
* Bangla-Bert was trained with code provided in Google BERT's github repository (https://github.com/google-research/bert)
* Currently released model follows bert-base-uncased model architecture (12-layer, 768-hidden, 12-heads, 110M parameters)
* Total Training Steps: 1 Million
* The model was trained on a single Google Cloud TPU 

## Evaluation Results

### LM Evaluation Results
After training 1 millions steps here is the evaluation resutls. 

```
global_step = 1000000
loss = 2.2406516
masked_lm_accuracy = 0.60641736
masked_lm_loss = 2.201459
next_sentence_accuracy = 0.98625
next_sentence_loss = 0.040997364
perplexity = numpy.exp(2.2406516) = 9.393331287442784
Loss for final step: 2.426227

```

### Downstream Task Evaluation Results
Huge Thanks to [Nick Doiron](https://twitter.com/mapmeld) for providing evalution results of classification task.
He used [Bengali Classification Benchmark](https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP) datasets for classification task.
Comparing to Nick's [Bengali electra](https://huggingface.co/monsoon-nlp/bangla-electra) and multi-lingual BERT, Bangla BERT Base achieves state of the art result.
Here is the [evaluation script](https://github.com/sagorbrur/bangla-bert/blob/master/notebook/bangla-bert-evaluation-classification-task.ipynb).


| Model | Sentiment Analysis | Hate Speech Task | News Topic Task | Average |
| ----- | -------------------| ---------------- | --------------- | ------- |
| mBERT | 68.15 | 52.32 | 72.27 | 64.25 |
| Bengali Electra | 69.19 | 44.84 | 82.33 | 65.45 |
| Bangla BERT Base | 70.37 | 71.83 | 89.19 | 77.13 |


**NB: If you use this model for any nlp task please share evaluation results with us. We will add it here.** 


## How to Use
You can use this model directly with a pipeline for masked language modeling:

```py
from transformers import BertForMaskedLM, BertTokenizer, pipeline

model = BertForMaskedLM.from_pretrained("sagorsarker/bangla-bert-base")
tokenizer = BertTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"????????? ?????????????????? {nlp.tokenizer.mask_token} ????????????"):
  print(pred)

# {'sequence': '[CLS] ????????? ?????????????????? ????????? ????????? ??? [SEP]', 'score': 0.13404667377471924, 'token': 2552, 'token_str': '?????????'}

```


## Author
[Sagor Sarker](https://github.com/sagorbrur)

## Acknowledgements

* Thanks to Google [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing the free TPU credits - thank you!
* Thank to all the people around, who always helping us to build something for Bengali.

## Reference
* https://github.com/google-research/bert





