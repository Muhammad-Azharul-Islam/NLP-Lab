# NLP Lab Programs

This repository contains implementations of various Natural Language Processing (NLP) programs for lab exercises.

## Programs List

1. **P1_text_scraping.py**: Text scraping from Wikipedia using Beautiful Soup library.
2. **P2_text_preprocessing.py**: Preprocessing of text including removal of special characters, URLs, stemming, lemmatization, and tokenization into words, sentences, and paragraphs. Counts vocabulary size, number of words, and sentences.
3. **P3_maximum_matching.py**: Implementation of the Maximum Matching algorithm for tokenization.
4. **P4_byte_pair_encoding.py**: Implementation of the Byte Pair Encoding algorithm for tokenization.
5. **P5_ngrams_tfidf.py**: Generation of unigrams, bigrams, trigrams, and TF-IDF using Python.
6. **P6_ngrams_tfidf_sklearn.py**: Generation of n-grams and TF-IDF using scikit-learn library.
7. **P7_naive_bayes_classification.py**: Naïve Bayes text classification on N-grams using scikit-learn and evaluation metrics.
8. **P8_word_cooccurrence_matrix.py**: Generation of word co-occurrence matrix for a given window size and calculation of similarity between word vectors.
9. **P9_word2vec.py**: Word2Vec embeddings using Gensim library with both pre-trained models and training on custom corpora.
10. **P10_IMDB_sentiment_LSTM_BERT.py**: IMDB sentiment analysis using Word2Vec embeddings and LSTM neural networks, as well as BERT language model.
11. **P11_IMDB_sentiment_BERT.py**: IMDB sentiment analysis using BERT language model and BERT classification.

## Requirements

The programs use various Python libraries including:

- beautifulsoup4
- nltk
- numpy
- scikit-learn
- pandas
- matplotlib
- gensim
- torch
- transformers
- seaborn
- tqdm

You can install all the required libraries using pip:

```bash
pip install beautifulsoup4 nltk numpy scikit-learn pandas matplotlib gensim torch transformers seaborn tqdm
```

## Usage

Each program can be run independently. For example:

```bash
python P1_text_scraping.py
```

## Program Descriptions

### P1: Text Scraping
Demonstrates web scraping from Wikipedia using the Beautiful Soup library.

### P2: Text Preprocessing
Implements various text preprocessing techniques such as removing special characters, tokenization, stemming, lemmatization, and provides statistics about the text.

### P3: Maximum Matching Algorithm
Implements the Maximum Matching algorithm for tokenization, which is a greedy approach that matches the longest possible word from a dictionary.

### P4: Byte Pair Encoding
Implements the Byte Pair Encoding algorithm, which is a data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.

### P5 & P6: N-grams and TF-IDF
Generate unigrams, bigrams, trigrams, and calculate TF-IDF scores for text documents, both with custom implementation and using scikit-learn.

### P7: Naïve Bayes Classification
Implements Naïve Bayes classifier for text classification using n-grams as features and evaluates the model performance.

### P8: Word Co-occurrence Matrix
Generates word co-occurrence matrix for different window sizes and calculates cosine similarity between word vectors.

### P9: Word2Vec Embeddings
Demonstrates Word2Vec using Gensim library, including using pre-trained models and training on custom corpora.

### P10 & P11: Sentiment Analysis
Implements sentiment analysis on IMDB movie reviews using different approaches:
- Word2Vec embeddings with LSTM neural networks
- BERT language model for feature extraction and classification 