# mnist-fastapi-compose

An example of MNIST training, eval and deployment with FastAPI and PyTorch and Docker Compose (Not for production usage)

```bash
docker compose build
```

```bash
docker compose run train
docker compose run evaluate
docker compose run --service-ports server
```

```bash
docker compose run infer
```
