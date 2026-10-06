FROM python:3.14-slim

# Hugging Face Spaces, container'ı UID 1000 ile çalıştırır
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# torch'un CPU sürümü önce kurulur (CUDA'lı sürüm GB'larca yer kaplar, model GPU'ya ihtiyaç duymuyor).
# 2.14.0+cpu, requirements.txt'teki torch==2.14.0 şartını karşıladığı için tekrar indirilmez.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .
COPY --chown=user templates/ templates/
COPY --chown=user models/ models/

# Render PORT ortam değişkenini atar, yoksa 8000 kullanır
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
