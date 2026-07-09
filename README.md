# Sentiment Analysis API

A production-ready Sentiment Analysis API developed during a six-week Applied NLP Engineering internship at Sohail Smart Solutions.

The project evolved from a traditional **TF-IDF baseline** into a **transformer-based sentiment analysis service** using **MiniLM sentence embeddings**, and was ultimately deployed, containerized, and monitored as a public REST API.

---

## Project Overview

This project classifies user text into three sentiment categories:

- Positive
- Negative
- Neutral

Rather than stopping at model training, the project follows the complete machine learning lifecycle, including:

- Text preprocessing
- Feature engineering
- Model comparison
- Error analysis
- REST API development
- Input validation
- Docker containerization
- Cloud deployment
- Production monitoring
- Technical documentation

The final system is deployed using **FastAPI**, **Docker**, and **Hugging Face Spaces**.

---

## Live Demo

### Public API

https://raghad232-sentiment-analysis-api.hf.space

### Swagger Documentation

https://raghad232-sentiment-analysis-api.hf.space/docs

### Health Endpoint

https://raghad232-sentiment-analysis-api.hf.space/health

### Metrics Endpoint

https://raghad232-sentiment-analysis-api.hf.space/metrics

---

## Features

- Transformer-based sentiment classification
- MiniLM sentence embeddings
- Logistic Regression classifier
- REST API built with FastAPI
- Input validation
- Structured logging
- Runtime monitoring
- Docker support
- Public cloud deployment
- Health monitoring endpoint
- Metrics endpoint
- Confidence score reporting
- Low-confidence detection

---

## System Architecture

The overall system architecture is illustrated in **docs/architecture.png**, showing the complete workflow from user input through preprocessing, transformer embeddings, prediction, monitoring, and API response.

---

## Technologies Used

- Python 3.11
- FastAPI
- scikit-learn
- Sentence Transformers
- all-MiniLM-L6-v2
- Logistic Regression
- Docker
- Hugging Face Spaces
- Joblib

---

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/Raghad-M461/sentiment-analysis-project.git

cd sentiment-analysis-project
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

## Example Request

```bash
curl -X POST \
"http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"text\":\"I absolutely love this product.\"}"
```

### Example Response

```json
{
  "label": "Positive",
  "confidence": 0.94,
  "low_confidence": false
}
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/predict` | Predict the sentiment of input text |
| `/health` | Verify that the service is running |
| `/metrics` | View monitoring statistics collected since startup |
| `/docs` | Interactive Swagger documentation |

---

## Model Evolution

Several feature representation techniques were evaluated throughout the project.

| Model | Accuracy | F1 Score |
|-------|----------:|----------:|
| TF-IDF | 69.5% | 68.2% |
| Averaged GloVe | 74.3% | 73.8% |
| MiniLM Sentence Embeddings | **82.4%** | **82.1%** |

The MiniLM sentence embedding model achieved the best overall performance and was selected as the final production model.

---

## Deployment Results

After deployment to Hugging Face Spaces:

- Successfully processed **16 live prediction requests**
- **0 runtime errors**
- Average response latency of **35.26 ms**
- Structured logging enabled
- Runtime monitoring through the `/metrics` endpoint
- Public REST API accessible through the live deployment

---

## Limitations

The current system has several known limitations:

- Optimized primarily for English-language text.
- Unsupported languages may produce unreliable predictions.
- Free cloud hosting introduces a short cold-start delay after inactivity.
- The training dataset is relatively small compared to commercial NLP datasets.

Future improvements include multilingual support, automatic language detection, larger training datasets, fine-tuning the transformer model, and integrating external monitoring platforms.

---

## Author

**Raghad Mohammad**

Applied NLP Engineering Internship

Sohail Smart Solutions
