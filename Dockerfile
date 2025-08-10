FROM python:3.12-slim-bookworm
LABEL authors="eloy"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx poppler-utils && \
    rm -rf /var/lib/apt/lists/*



ENV PATH="/usr/local/bin:${PATH}"
ENV OLLAMA_API_URL="http://host.docker.internal:11434"
ENV HF_HOME=/root/.cache/huggingface


ARG MODEL_ID_1="Qwen/Qwen3-0.6B"
ARG MODEL_ID_2="ds4sd/docling-models"


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Comment following line if your machine has GPU acceleration
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#RUN python -c "from huggingface_hub import snapshot_download; \
#    snapshot_download(repo_id='${MODEL_ID_1}', \
#    allow_patterns=['*.bin', '*.json', '*.txt', '*.model', '*.safetensors'])"

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${MODEL_ID_2}', \
    allow_patterns=['*.bin', '*.json', '*.txt', '*.model', '*.safetensors'])"


RUN python -c "\
import easyocr; \
reader = easyocr.Reader(['en', 'es']); \
reader.readtext('test.png', detail=0, paragraph=False) if False else None"


COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false"]
