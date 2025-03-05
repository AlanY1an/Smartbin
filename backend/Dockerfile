# Use GPU image if needed, otherwise use python:3.10
ARG USE_GPU=false
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu
FROM python:3.10 AS cpu

# Select base image based on ARG
FROM ${USE_GPU:+gpu} AS final

# Set working directory inside backend 
WORKDIR /backend

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files from backend folder
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI application
CMD ["python", "main.py"]
