FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates
COPY app.py ./
COPY store_index.py ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]