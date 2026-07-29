FROM python:3.12-bookworm

WORKDIR /app

COPY pyproject.toml main.py pipeline.py best_model history.csv ./

RUN pip install uv

RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app"]
