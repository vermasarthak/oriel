FROM python:3.11-slim

WORKDIR /app

# Install standard dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy source
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

ENV ORIEL_DB_PATH=/data/oriel.db
ENV ORIEL_API_KEY=your-secret-token

VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "oriel.api:app", "--host", "0.0.0.0", "--port", "8000"]
