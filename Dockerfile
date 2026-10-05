FROM ghcr.io/astral-sh/uv:0.12.16 AS uv

FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/

WORKDIR /code

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY artifacts ./artifacts
COPY main.py ./

EXPOSE 80

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "80"]
