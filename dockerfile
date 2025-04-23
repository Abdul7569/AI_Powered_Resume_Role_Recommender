FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    git \
    poppler-utils \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Preload models
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Generate embeddings inside image
COPY job_title_des.csv .
RUN python -c "\
import pandas as pd, pickle; \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('all-mpnet-base-v2'); \
desc = pd.read_csv('job_title_des.csv')['Cleaned_Description'].tolist(); \
embs = model.encode(desc, convert_to_tensor=True); \
pickle.dump(embs, open('role_embeddings.pkl', 'wb'))"

# Now copy the rest of the code
COPY . .

EXPOSE 8501
RUN mkdir -p /app/artifacts /app/logs
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
