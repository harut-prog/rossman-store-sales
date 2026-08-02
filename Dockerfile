FROM python:3.12-bookworm
WORKDIR /app

ENV PYTHONWARNINGS=ignore

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-default-groups --group deploy

COPY main.py pipeline.py ./
COPY best_model ./best_model
COPY templates ./templates
COPY static ./static

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
