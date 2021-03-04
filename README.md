# Task-Stylumia
Task-Stylumia is an assignment report for Data Science Internship program at Stylumia. In order to progress further with the screening process for the Data Science internship, I need to solve the [StumbleUpon Evergreen Classification Challenge](https://www.kaggle.com/c/stumbleupon/), which needs to be submitted by 1st March 2021.

## Submission guidelines

- Link to a Source code (Preferably on GitHub)
- CSV file containing your result (as described above for submit.csv)
- File describing your approach of solving the problem. Include precision and recall for each class.

## Method
For the problem I preprocessed the dataset by parsing through the files provided in the dataset. After a subjective analysis I found that we can extract additional textual data by parsing the files, for the quick rush I managed to extract title and body from the files.
Next, I filled empty data attributes and selected relevant features from the dataset. The dataset was noisy so I normalized it using sklearn's Normalizer.
For textual data, I combined the `alchemy_category`, `title` and the `body` with additional special tokens to distinguish between them. 

## Deep learning Model
For NLP task with embeddings, I found torchText complex and it consumed my most of the time so I chose [Pytorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/) for embeddings task. 
Using `StaticTokenizerEncoder` and `pad_tensor` from `torchnlp.encoders.text`, I tokenized the texts and padded as well.

### Bi-Linear Network:
I modeled a bilinear network, comprising of a textual model and a sequential-dense model.

Textual Model:
- Embedding Layer containing embeddings from GloVE
- Transformer Encoder with 6 attention heads and 3 attention layers
- Transformer Encoder with 5 attention heads and 3 attention layers
- Dense Layer + Batch Normalization Layer

Sequential-Dense Model:
- Linear Layer of input_size = 21 and output_size = 128, and ReLU activation
- Linear Layer of input_size = 128 and output_size = 64, and ReLU activation
- Droupout with probabilty ratio of 0.4
- Linear Layer of input_size = 64 and output_size = 32, and ReLU activation

Concatenation:
- Concatenated using a BiLinear layer from `torch.nn`, of input_size = (32, 256) and output_size = 128
- Droupout with probabilty ratio of 0.35
- Linear Layer of input_size = 128 and output_size = 32
- 1D Batch Normalization layer for 32-size tensor input 
- Classifier layer of input_size = 32 and output_size = 1

# Cite
StumbleUpon Evergreen Classification Challenge: `https://www.kaggle.com/c/stumbleupon/`

Maarten Bosma approach: `https://github.com/ma2rten/kaggle-evergreen`

Python BoilerPipe: `https://github.com/misja/python-boilerpipe`

GloVe:
```
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}
```

Transformers: 
```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, 
          Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, 
          Lukasz and Polosukhin, Illia},
  journal={arXiv preprint arXiv:1706.03762},
  year={2017}
}
```
