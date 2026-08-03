FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY study_assistant ./study_assistant
COPY data ./data

RUN uv sync --locked --no-dev

RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN uv run python -m study_assistant.ingest

EXPOSE 5000

CMD ["python", "-m", "study_assistant.app"]