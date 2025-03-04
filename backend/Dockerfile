# If use GPU, use this line
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


WORKDIR /app


RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Fast API to Docekr container 
COPY . .

# Run Fast API server
CMD ["python", "main.py"]
