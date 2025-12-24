FROM python:3.11-slim

RUN pip install --upgrade pip

RUN pip install fastapi==0.121.2
RUN pip install uvicorn==0.38.0
RUN pip install pydantic==2.12.4
# Install CPU-only PyTorch (no CUDA)
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install diffusers==0.35.2
#RUN pip install transformers==4.57.1
#RUN pip install accelerate==1.11.0
#RUN pip install Pillow==12.0.0
#RUN pip install fastapi-limiter==0.1.6
RUN pip install aiosqlite==0.21.0
RUN pip install peft==0.18.0

RUN rm -rf /root/.cache/pip

WORKDIR /app

COPY server.py /app/server.py

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]

