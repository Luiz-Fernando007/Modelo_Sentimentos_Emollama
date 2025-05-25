FROM python:3.10

WORKDIR /api_emollama

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use --reload apenas para desenvolvimento
CMD ["python" ,"consumer.py"]