# SPS GenAI Assignment 1 API

This FastAPI project provides two endpoints:

- `POST /generate` generates text with a simple bigram model.
- `POST /embedding` returns the 300-dimensional `en_core_web_lg` spaCy
  embedding for a query word.

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
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

Then open <http://127.0.0.1:8000/docs> or query the endpoint with `curl`.
