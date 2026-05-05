FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates
COPY app.py ./

RUN pip install --no-cache-dir --upgrade pip

# CPU-only PyTorch avoids huge CUDA/NVIDIA downloads
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
