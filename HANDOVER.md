# Project Handover

## Overview

This repository contains the complete sentiment analysis project developed during my internship at Sohail Smart Solutions as part of ENGR 391 at the American University of Ras Al Khaimah.

The project evolved from a classical machine learning prototype into a complete AI engineering project through multiple development stages, including text preprocessing, feature engineering, contextual embeddings, transformer fine-tuning, REST API development, cloud deployment, automated testing, runtime monitoring, evaluation, and documentation.

The final deployed solution uses a fine-tuned DistilBERT model for sentiment classification and demonstrates the complete lifecycle of developing, deploying, evaluating, and maintaining an AI service.

---

# Live Service

## Deployment

**Hugging Face Spaces**

https://huggingface.co/spaces/Raghad232/sentiment-analysis-api

### Example Request

**POST** `/predict`

```json
{
  "text": "The product exceeded my expectations."
}
```

### Example Response

```json
{
  "label": "Positive",
  "confidence": 0.97
}
```

### Additional Endpoints

Health Check

```
GET /health
```

Monitoring

```
GET /metrics
```

---

# Repository Structure

## .github/workflows/

Contains the GitHub Actions workflow used to automate testing and continuous integration.

## app/

Contains project documentation related to inference and API usage.

## embeddings/

Contains experiments and resources related to GloVe, MiniLM, and contextual sentence embeddings.

## finetune/

Contains documentation related to the transformer fine-tuning stage of the project.

## evaluation/

Contains the Sohail AI Evaluation Framework used to evaluate AI systems across Data, Model, Evaluation, Service & Deployment, and Responsible AI.

---

# Main Python Files

**preprocessing.py**

Implements the text preprocessing pipeline.

**train_and_save.py**

Trains and saves the sentiment analysis models.

**sentiment_analysis.py**

Implements the baseline sentiment analysis workflow.

**sentence_features_comparison.py**

Compares TF-IDF, GloVe, and MiniLM sentence representations.

**contextual_similarity.py**

Evaluates contextual similarity using transformer embeddings.

**predict.py**

Runs sentiment predictions using the trained model.

**api_app.py**

Implements the FastAPI REST API used for deployment.

**error_analysis.py**

Analyzes model errors and misclassified samples.

**test_predictions.py**

Runs regression tests to verify prediction consistency.

---

# Project Documentation

The repository includes:

- README.md
- CASE_STUDY.md
- docs/architecture.png
- evaluation/FRAMEWORK.md
- FINAL_COMPARISON.md
- finetune/limitations.md

These documents describe the complete development process, deployment architecture, evaluation methodology, and final project recommendations.

---

# Fine-Tuning Capstone

The final stage of the internship introduced end-to-end fine-tuning using the pretrained DistilBERT model.

## Final Comparison

| Model | Accuracy | Macro-F1 |
|--------|---------:|---------:|
| TF-IDF + Logistic Regression | 59.38% | 55.33% |
| MiniLM + Logistic Regression | 78.13% | 77.22% |
| Fine-Tuned DistilBERT | **84.38%** | **84.24%** |

## Final Recommendation

The fine-tuned DistilBERT model is recommended for deployment because it achieved the highest overall predictive performance while demonstrating the strongest understanding of contextual language and negation.

MiniLM remains an excellent lightweight alternative for applications where lower latency or smaller model size is more important than maximum predictive performance.

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn api_app:app --reload
```

After the server starts, open:

```
http://127.0.0.1:8000/docs
```

to access the interactive Swagger API documentation.

---

# Maintenance Guide

For future development and maintenance:

- Retrain the model after adding new labeled data.
- Run regression tests before deployment.
- Re-evaluate the project using the AI Evaluation Framework.
- Update FINAL_COMPARISON.md whenever a new model is introduced.
- Verify the deployed Hugging Face service after updates.
- Keep project documentation synchronized with implementation changes.

---

# Known Limitations

- Small labeled dataset.
- English-focused training and evaluation.
- Limited testing on unseen domains.
- Single fine-tuning experiment.
- Basic runtime monitoring.
- No authentication or user management.

---

# Suggested Future Improvements

- Increase dataset size.
- Support multilingual sentiment analysis.
- Fine-tune multilingual transformer models.
- Add automatic language detection.
- Build a real-time monitoring dashboard.
- Implement authentication and authorization.
- Introduce continuous retraining pipelines.
- Improve confidence calibration.

---

# Repository Status

The repository contains:

- Source code
- Model training pipeline
- FastAPI application
- Cloud deployment
- Automated testing
- Runtime monitoring
- Case study
- Architecture documentation
- AI Evaluation Framework
- Fine-tuning comparison
- Final handover documentation

The project is ready for future maintenance and further development.

---

# Internship Summary

Throughout the internship, the project progressed through the complete AI engineering lifecycle:

- Data preprocessing
- Dataset expansion
- Feature engineering
- Classical machine learning
- Word embeddings
- Contextual sentence embeddings
- REST API development
- Docker containerization
- Cloud deployment
- System testing
- Transformer fine-tuning
- AI evaluation
- Documentation and handover

This repository serves as the final deliverable of the internship and provides a complete reference for future engineers who maintain or extend the project.
