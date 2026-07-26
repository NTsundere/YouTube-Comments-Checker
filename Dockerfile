FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scikit-learn==1.6.1

RUN python -m nltk.downloader stopwords wordnet

CMD ["python3", "app/app.py"]