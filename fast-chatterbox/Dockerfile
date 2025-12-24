FROM python:3.11-slim

RUN apt-get update && apt-get install -y git libsndfile1 ffmpeg build-essential curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install "git+https://github.com/travisvn/chatterbox-multilingual.git@exp"

RUN mkdir -p /root/.cache/chatterbox

WORKDIR /app
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

