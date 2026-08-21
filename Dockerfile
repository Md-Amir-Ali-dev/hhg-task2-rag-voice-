FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies if required (mostly for faiss/fastembed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed model to cache it during build
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup'])); print('fastembed model cached OK')"

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE $PORT

# Run the server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
