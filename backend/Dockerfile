# Allow switching between GPU and CPU images
ARG USE_GPU=false

# GPU base image (NVIDIA CUDA)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# CPU base image (Python 3.10)
FROM python:3.10 AS cpu

# Select final base image based on the USE_GPU argument
FROM ${USE_GPU:+gpu} AS final

# Set working directory inside container
WORKDIR /backend  

# Ensure pip is available
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend project
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI application
CMD ["python", "main.py"]
