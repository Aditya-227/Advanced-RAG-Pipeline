FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU first
RUN pip install --no-cache-dir torch==2.9.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create data directories
RUN mkdir -p /app/data/faiss_index \
             /app/data/bm25_index \
             /app/data/uploaded_pdfs \
             /app/data/metrics \
             /app/data/hf_cache

# Set cache to local folder (persistent in HF Spaces)
ENV HF_HOME=/app/data/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/hf_cache
ENV DATA_DIR=/app/data
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
