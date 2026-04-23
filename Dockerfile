FROM python:3.11-slim

WORKDIR /app

# System deps for audio + postgres client
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dep resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first for cacheable layer
COPY pyproject.toml ./
RUN uv pip install --system --no-cache .

# Copy source
COPY src/ ./src/
COPY data/ ./data/

# Runtime
ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "-m", "src.voice.server"]
