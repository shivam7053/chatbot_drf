# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Install System Dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Project Files ----------
COPY . .

# ---------- Install Python Dependencies ----------
RUN pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install spacy==3.6.1
RUN pip install -r requirements.txt

# ---------- Download SpaCy Model ----------
RUN python -m spacy download en_core_web_md --direct

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run Django via Gunicorn ----------
CMD ["gunicorn", "chatbot.wsgi:application", "--bind", "0.0.0.0:8000"]
