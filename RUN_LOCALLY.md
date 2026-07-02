# Running the API Locally (without Docker)

If Docker is not available, the project can be run using a Python virtual environment.

## Requirements

- Python 3.11
- Git

## Setup

```bash
# Clone the repository
git clone https://github.com/Raghad-M461/sentiment-analysis-project.git
cd sentiment-analysis-project

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 stopwords

# Train the model (run once)
python train_and_save.py

# Start the API
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

## Test the API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"not bad, actually quite impressed"}'
```

## Note

Docker was not available on my machine, so I tested the application using this virtual environment setup. The project also includes a Dockerfile for containerized deployment, and GitHub Actions was used to verify that the API works correctly.
