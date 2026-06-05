# Sentiment Analysis Project

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify customer feedback as either positive or negative.

The model uses TF-IDF vectorization and Logistic Regression to learn patterns from customer reviews and predict sentiment for new text.

## Dataset

The dataset contains 80 manually labelled customer feedback sentences:

* 40 Positive examples
* 40 Negative examples

## Repository Files

* `sentiment_analysis.py` – Original sentiment analysis model
* `ngram_comparison.py` – N-gram enhancement and evaluation experiment
* `evaluation/README.md` – Sohail AI Evaluation Framework documentation

## N-Gram Enhancement Experiment

As part of the project enhancement phase, three n-gram configurations were evaluated:

| Configuration | N-Gram Range                        |
| ------------- | ----------------------------------- |
| Config 1      | Unigrams (1,1)                      |
| Config 2      | Unigrams + Bigrams (1,2)            |
| Config 3      | Unigrams + Bigrams + Trigrams (1,3) |

The goal was to investigate whether including longer word combinations would improve sentiment classification, particularly for phrases containing negation such as "not good" and "not bad".

## Evaluation Method

The models were evaluated using:

* TF-IDF Vectorization
* Logistic Regression
* 5-Fold Cross-Validation

A separate negation test was also performed using challenging sentiment examples.

## Results

| Configuration                       | Accuracy |
| ----------------------------------- | -------- |
| Unigrams (1,1)                      | 80.0%    |
| Unigrams + Bigrams (1,2)            | 73.8%    |
| Unigrams + Bigrams + Trigrams (1,3) | 75.0%    |

## Key Findings

The unigram model achieved the highest overall accuracy on this dataset.

Although bigrams and trigrams were expected to improve performance, the dataset is relatively small, which limited their effectiveness. The additional word combinations increased the number of features without providing enough training examples for the model to learn useful patterns.

Negation testing also revealed an important limitation. Words such as "not", "good", and "bad" were not present in the training dataset, meaning the model could not truly learn their relationships. This highlighted the importance of dataset quality and vocabulary coverage when evaluating sentiment analysis systems.
