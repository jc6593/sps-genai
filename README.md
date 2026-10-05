# SPS Generative AI API

This FastAPI project provides three endpoints:

- `POST /generate` generates text with a simple bigram model.
- `POST /embedding` returns the 300-dimensional `en_core_web_lg` spaCy
  embedding for a query word.
- `POST /classify` classifies an uploaded image as one of the ten CIFAR-10
  classes.

## Local setup

```bash
uv sync
uv run fastapi dev main.py
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

Example embedding request:

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H 'Content-Type: application/json' \
  -d '{"query_word":"apple"}'
```

The response contains the normalized query, vector dimension, and full vector:

```json
{
  "query_word": "apple",
  "dimension": 300,
  "embedding": [-0.36391, 0.43771, -0.20447]
}
```

The example above abbreviates the vector; the API returns all 300 values.

## Docker

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 8000:80 sps-genai-assignment2
```

Then open <http://127.0.0.1:8000/docs> or query the endpoint with `curl`.

## Assignment 2: CIFAR-10 CNN

The Assignment 2 model follows the required 64 x 64 architecture:

```text
3 x 64 x 64
-> Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
-> Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
-> Flatten -> Linear(8192, 100) -> ReLU -> Linear(100, 10)
```

Train and evaluate the classifier on CIFAR-10:

```bash
uv run python train_cifar10.py --epochs 10 --batch-size 256
```

The training script resizes CIFAR-10 images to 64 x 64 and saves the trained
checkpoint to `artifacts/cifar10_cnn.pt`.

The committed checkpoint was trained for 10 epochs and achieved **60.27% test
accuracy** on all 10,000 CIFAR-10 test images after being reloaded from disk.

After training, start the API and classify an image:

```bash
uv run fastapi dev main.py
curl -X POST http://127.0.0.1:8000/classify \
  -F 'image=@example.png'
```

The response contains the predicted class, class index, confidence, and the
probability assigned to every CIFAR-10 class.

Example response:

```json
{
  "predicted_class": "cat",
  "class_index": 3,
  "confidence": 0.6251,
  "probabilities": {
    "airplane": 0.0003,
    "automobile": 0.0001,
    "bird": 0.1253,
    "cat": 0.6251,
    "deer": 0.0485,
    "dog": 0.1942,
    "frog": 0.0001,
    "horse": 0.0062,
    "ship": 0.0001,
    "truck": 0.0001
  }
}
```

To verify the pipeline quickly before a full training run:

```bash
uv run python train_cifar10.py \
  --epochs 1 \
  --max-train-batches 2 \
  --max-test-batches 2 \
  --checkpoint /tmp/cifar10_cnn_smoke.pt
```
