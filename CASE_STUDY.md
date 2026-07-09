# Case Study: Building a Production-Ready Sentiment Analysis API

## Overview

This project began as a traditional sentiment analysis task and gradually evolved into a complete, production-ready machine learning service. Over six weeks, the project progressed from experimenting with basic text representation techniques to deploying and monitoring a transformer-based sentiment analysis API on the cloud. The objective was not only to improve prediction quality but also to build a reliable system that could be accessed, monitored, and maintained like a real-world application.

## The Problem

The goal of the project was to classify user text as **Positive**, **Negative**, or **Neutral**. Although sentiment analysis is a common Natural Language Processing (NLP) task, creating a reliable service requires much more than training a machine learning model. The project needed to address data preprocessing, model selection, evaluation, deployment, monitoring, and documentation.

## Technical Approach

The project started with a traditional **TF-IDF** representation combined with a Logistic Regression classifier. While this baseline provided reasonable performance, it struggled to understand sentence meaning, especially when negation or different wording changed the sentiment.

To improve the model, word and sentence embedding techniques were explored. These approaches captured semantic relationships between words better than TF-IDF and produced more meaningful text representations.

The final solution used **MiniLM transformer sentence embeddings**, which provide contextual representations of complete sentences. These embeddings were combined with a Logistic Regression classifier, resulting in more accurate and consistent predictions while remaining lightweight enough for cloud deployment.

Throughout development, text preprocessing was carefully designed to clean and normalize user input before generating embeddings and making predictions.

## Error Analysis

Rather than relying only on evaluation metrics, the project included a detailed error analysis phase. Misclassified examples were manually reviewed to understand the model's weaknesses.

This analysis showed that traditional TF-IDF features often struggled with negation, ambiguous wording, and unseen sentence structures. Additional testing also demonstrated that unsupported languages such as Arabic could produce confident but unreliable predictions because the model was trained primarily on English data.

These findings directly influenced later improvements, including the adoption of transformer embeddings and the addition of a low-confidence indicator in the API responses.

## Deployment and Monitoring

Once the final model was selected, it was converted into a reusable REST API using **FastAPI**. The service exposes endpoints for prediction, health checking, and monitoring.

The project was containerized using Docker and deployed to **Hugging Face Spaces**, allowing the API to be accessed through a public URL.

To improve reliability, structured logging was implemented for every prediction request. The deployed service records timestamps, request IDs, input length, prediction confidence, response latency, and low-confidence predictions. A dedicated `/metrics` endpoint reports request statistics, error counts, average latency, and prediction distribution since startup.

## Results

The deployed service successfully processed **16 live prediction requests** during monitoring tests with **0 errors**. The average response latency was approximately **35.26 ms**, demonstrating that the API remained responsive after deployment. Monitoring also recorded **5 low-confidence predictions (31.25%)**, providing useful insight into uncertain or out-of-domain inputs.

## Limitations and Future Work

The current system is primarily designed for English text and may produce unreliable predictions for unsupported languages. The Hugging Face deployment also experiences short cold starts after periods of inactivity, which is common on free hosting platforms.

Future improvements could include multilingual support, automatic language detection, confidence calibration, larger training datasets, and replacing the Logistic Regression classifier with an end-to-end transformer fine-tuning approach. These improvements would further increase both prediction quality and production readiness.
