# Deployment Notes

## Local vs Cloud Deployment

Running the API locally means it is only accessible on my own computer through localhost.

Deploying to a cloud platform makes the API accessible through a public URL so anyone can use it over the internet.

Cloud deployment also requires:
- A public URL
- A start command
- Listening on the port provided by the hosting platform
- Installing all required dependencies

---

## Free Hosting Platform Requirements

For deployment on Hugging Face Spaces (Docker), the application needs:

- A Dockerfile
- A requirements.txt file
- Pinned Python dependencies
- A command to start the API
- The application must listen on the correct port

---

## Model File

The trained model is included with the deployment.

Keeping the model inside the deployment makes startup faster because it does not need to be downloaded every time the application starts.

The trade-off is a slightly larger repository and deployment size.

---

## Deployment Result

Platform:
Hugging Face Spaces (Docker)

Live API:

https://raghad232-sentiment-analysis-api.hf.space

Health Check:

GET /health

Response:

```json
{
  "status": "ok"
}
```

The deployed API was successfully tested using the `/predict` endpoint with multiple sample requests and returned valid predictions.
