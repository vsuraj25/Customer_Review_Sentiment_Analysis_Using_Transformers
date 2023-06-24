# Sentiment Analysis on Amazon Fine Food Reviews

This project focuses on performing sentiment analysis on the Amazon Fine Food Reviews dataset using two different approaches: VADER (Valence Aware Dictionary and Sentiment Reasoner) with a Bag of Words approach and the Roberta Pretrained Model from Transformers.

## Dataset

The dataset used for this project is the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews). It contains reviews of fine foods from Amazon, including a variety of information such as the product ID, reviewer ID, helpfulness rating, review text, and more. The dataset provides a rich source of data for sentiment analysis tasks.

## Approaches

### VADER (Valence Aware Dictionary and Sentiment Reasoner) - Bag of Words approach

VADER is a lexicon-based approach for sentiment analysis that is specifically tuned for social media texts. It uses a combination of lexical and grammatical heuristics to determine the sentiment of a given text. In this project, we employed the Bag of Words approach, which represents text data as a bag of individual words, disregarding grammar and word order. The VADER sentiment analyzer assigns sentiment scores to each review, indicating the positivity, neutrality, or negativity of the text.

### Roberta Pretrained Model from Transformers

RoBERTa is a robustly optimized method for pretraining natural language processing (NLP) models. It builds upon the popular BERT architecture and achieves state-of-the-art performance on various NLP tasks. In this project, we utilized the Roberta pretrained model from the Transformers library. The model was fine-tuned on the Amazon Fine Food Reviews dataset for sentiment analysis. By feeding review texts into the model, we obtained sentiment predictions for each review.

## Results

The sentiment analysis results from both approaches were compared and evaluated based on accuracy, precision, recall, and F1-score. The performance of each approach was analyzed to understand their strengths and limitations in sentiment analysis tasks.

## Usage

To reproduce the sentiment analysis results, follow these steps:

1. Download the Amazon Fine Food Reviews dataset from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).
2. Preprocess the dataset by cleaning and preparing the review texts.
3. Implement the VADER approach using a Bag of Words representation and calculate sentiment scores for each review.
4. Fine-tune the Roberta Pretrained Model using the Transformers library on the dataset.
5. Feed the review texts to the fine-tuned Roberta model and obtain sentiment predictions.
6. Evaluate the performance of both approaches by comparing the sentiment analysis results.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- NLTK (Natural Language Toolkit)
- Transformers library
