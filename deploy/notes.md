# Deployment Notes — Taking the API Live

**Branch:** `feature/cloud-deploy`  
**Platform:** Hugging Face Spaces (free tier, Docker SDK)  
**Public URL:** https://raghad-m461-sentiment-api.hf.space

---

## Part 1 — Concepts

### Running locally vs deploying to a hosting platform

When the API runs locally or in GitHub Actions, it only exists on that machine
and is only reachable from that machine. Anyone else who wants to use it would
need to clone the repo, install the dependencies, train the model, and start
the server themselves. That is not a usable service.

A hosting platform runs the application on a server that has a permanent
internet connection and a public IP address. The platform assigns a URL —
something like `https://your-app-name.hf.space` — and routes any HTTP request
sent to that URL to your running application. Anyone in the world with that URL
can send a request and get a response.

Three things that matter on a hosting platform but not locally:

**Public URL:** The platform provides this. For Hugging Face Spaces it is
automatically `https://<username>-<space-name>.hf.space`. Nothing needs to be
configured; the URL exists as soon as the Space is created.

**Environment variables:** A hosting platform does not know what port the
application wants to listen on. Instead of hardcoding `8000`, the application
should read the port from the `PORT` environment variable that the platform
sets at runtime. In the Dockerfile this is handled with:
```
CMD uvicorn api_app:app --host 0.0.0.0 --port ${PORT:-7860}
```
The `${PORT:-7860}` means "use whatever PORT the platform sets, and fall back
to 7860 if it is not set". Hugging Face Spaces expects port 7860.

**Start command:** The hosting platform needs to know how to start the
application. With a Dockerfile, the `CMD` instruction serves this role. The
platform builds the image and then runs whatever CMD specifies.

### What a free-tier host needs from you

Every free hosting platform needs essentially the same three things:

1. **A start command** — either in the Dockerfile `CMD`, a `Procfile`, or
   a platform-specific configuration file. For Hugging Face Spaces with Docker,
   the `CMD` in the Dockerfile is used directly.

2. **Port from an environment variable** — the platform sets `PORT` and the
   application must listen on it. Hardcoding `8000` would mean the app listens
   on a port the platform does not expose publicly.

3. **Pinned dependencies** — a `requirements.txt` with exact versions ensures
   the platform installs the same packages that were tested locally and in CI.
   Without pinning, `pip install fastapi` might install a different version
   tomorrow that breaks the app in a way that is hard to diagnose.

### Why the model must ship with the deploy — and the trade-off

The hosting platform starts from the Dockerfile. At startup it has no knowledge
of the trained Logistic Regression model — it only has whatever was built into
the image. There are two approaches:

**Option 1: Bake the trained model into the image.** Run `train_and_save.py`
during the Docker build step (`RUN python train_and_save.py`). The
`logreg_classifier.joblib` file is then part of the image. Startup is fast
because the model is already there. The downside is a slightly larger image
and the need to rebuild the image whenever the model is retrained.

**Option 2: Download the model at startup.** The container starts, then
downloads the model from an external storage location (S3, HuggingFace Hub,
GitHub). Startup is slow on the first request (cold start) but the image is
smaller and the model can be updated without rebuilding the image.

For this project, Option 1 was used: the Logistic Regression classifier is
trained and saved during the Docker build step. The sentence-transformer
encoder (`all-MiniLM-L6-v2`) is NOT baked in because at ~90MB it would make
the image much larger and it does not change between deployments. It is
downloaded from HuggingFace at first startup, which is the main source of
cold-start latency on the live platform.

---

## Part 2 — Deployment

### Platform choice: Hugging Face Spaces

Render's free tier has a 512MB RAM limit, which is too small for PyTorch and
sentence-transformers. Railway gives $5 of free credit. Hugging Face Spaces
has a free CPU tier with 16GB disk and 16GB RAM, and is specifically designed
for ML model serving — making it the best fit for this project.

### How to deploy

```bash
# 1. Create a new Space at https://huggingface.co/new-space
#    - Owner: your HF username
#    - Space name: sentiment-api
#    - SDK: Docker
#    - Visibility: Public

# 2. Clone the Space repo
git clone https://huggingface.co/spaces/Raghad-M461/sentiment-api
cd sentiment-api

# 3. Copy the project files into the Space repo
cp /path/to/sentiment-analysis-project/Dockerfile .
cp /path/to/sentiment-analysis-project/requirements.txt .
cp /path/to/sentiment-analysis-project/api_app.py .
cp /path/to/sentiment-analysis-project/predict.py .
cp /path/to/sentiment-analysis-project/preprocessing.py .
cp /path/to/sentiment-analysis-project/train_and_save.py .
mkdir -p embeddings
cp /path/to/sentiment-analysis-project/embeddings/sentiment_dataset_enriched.csv embeddings/

# 4. Push to HuggingFace — this triggers the build
git add .
git commit -m "initial deployment"
git push
```

The Space then builds the Docker image (takes ~10–15 minutes the first time
because of PyTorch and sentence-transformers), and the API becomes live at:

```
https://raghad-m461-sentiment-api.hf.space
```

### Live test results

**Cold start (first request after idle):** ~25–35 seconds  
This is the time for the sentence-transformer encoder to download from
HuggingFace (~90MB). Subsequent requests on the same warm instance take
under 200ms.

**Warm request latency:** ~120–180ms  
The extra latency compared to local testing (~2ms) is network round-trip time
plus the overhead of the Spaces infrastructure routing the request.

#### 5 live requests to the deployed API

**Test 1 — Clear positive:**
```bash
curl -X POST https://raghad-m461-sentiment-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The product quality is outstanding and the delivery was fast"}'
```
```json
{"label":"Positive","confidence":0.7812,"low_confidence":false}
```

**Test 2 — Negation case:**
```bash
curl -X POST https://raghad-m461-sentiment-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This is not good at all, very disappointed with the service"}'
```
```json
{"label":"Negative","confidence":0.5712,"low_confidence":false}
```

**Test 3 — Mixed sentiment:**
```bash
curl -X POST https://raghad-m461-sentiment-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The app works fine but the customer support could be better"}'
```
```json
{"label":"Negative","confidence":0.5401,"low_confidence":true}
```

**Test 4 — Negation of negative:**
```bash
curl -X POST https://raghad-m461-sentiment-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Not bad, actually quite impressed with how well it performs"}'
```
```json
{"label":"Positive","confidence":0.6647,"low_confidence":false}
```

**Test 5 — Non-English (Arabic) — out-of-domain case:**
```bash
curl -X POST https://raghad-m461-sentiment-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "هذا المنتج رائع جدا"}'
```
```json
{"label":"Neutral","confidence":0.616,"low_confidence":false}
```

Note on Test 5: the Arabic input returns a plausible-looking response with
confidence above the low_confidence threshold. As documented in the validation
task, this is a known limitation — the model was not trained on Arabic text
and the label carries no real meaning. Language detection at the API boundary
is the next hardening step.

### Cold start vs warm request

| Request type | Latency |
|---|---|
| Cold start (encoder download) | 25–35 seconds |
| Warm request (model loaded) | 120–180 ms |

The cold start time is entirely from downloading the all-MiniLM-L6-v2 encoder
weights. On Hugging Face Spaces free tier, the instance idles and shuts down
after ~15 minutes of inactivity, so the next request after a period of no
traffic will trigger a cold start. This is normal behaviour for free-tier
hosting and is documented here so anyone using the API knows what to expect.

### Health check

```bash
curl https://raghad-m461-sentiment-api.hf.space/health
```
```json
{"status":"ok"}
```
