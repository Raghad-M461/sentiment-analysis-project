# Docker Notes

## What is a container?

A container is an isolated environment that includes the application, its dependencies, and the required runtime. Unlike running Python locally, where the application depends on the software installed on the computer, a container provides the same environment on any machine, helping avoid the "works on my machine" problem.

## Dockerfile, Image, and Container

- **Dockerfile:** A text file that contains the instructions for building the application environment.
- **Image:** The packaged result created from a Dockerfile. It contains the application, dependencies, and runtime.
- **Container:** A running instance of a Docker image. Multiple containers can be created from the same image while remaining isolated from each other.

## Why pin dependencies?

Pinning dependencies means specifying the exact version of each package in `requirements.txt`. This ensures everyone installs the same versions, making the project consistent and reproducible across different machines.

## EXPOSE and CMD

`EXPOSE 8000` indicates that the application listens on port 8000 inside the container. When running the container, the port is mapped using:

```bash
docker run -p 8000:8000 sentiment-api
```

The `CMD` instruction starts the FastAPI application:

```dockerfile
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Using `0.0.0.0` allows the API to be accessed from outside the container.

---

## Docker Availability

Docker was not available on my local machine, so I followed the alternative option allowed in the assignment. I prepared the Dockerfile, pinned `requirements.txt`, and `RUN_LOCALLY.md` for the virtual environment setup. The application was tested successfully using GitHub Actions, while the Docker commands below are ready to use on any machine with Docker installed.

### Build the image

```bash
docker build -t sentiment-api .
```

### Run the container

```bash
docker run -p 8000:8000 sentiment-api
```

### Run without Docker

```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```
