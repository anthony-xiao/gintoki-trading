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


# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# # System setup
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     nvidia-cuda-toolkit \
#     && rm -rf /var/lib/apt/lists/*

# # Python environment
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# RUN pip install nvidia-cudnn-cu12==8.9.4.25

# # Application code
# WORKDIR /app
# COPY . .

# # Enable GPU acceleration
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

# CMD ["python", "-m", "ml_core.training.train"]
