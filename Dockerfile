FROM python:3.11-slim 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src/py"

CMD ["python", "-m", "data_ingestion.historical_data_fetcher"]
