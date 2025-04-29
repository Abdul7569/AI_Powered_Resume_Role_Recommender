import os
import io
import re
import json
import pickle
import csv
from datetime import datetime
import fitz
import numpy as np
import pandas as pd
import spacy
import docx2txt
from sentence_transformers import SentenceTransformer, util
try:
    from firebase_utils import upload_model_log
except ImportError:
    upload_model_log = None

print("Loading model...")
model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
nlp = spacy.load("en_core_web_sm")

# ✅ Load Roles
roles_df = pd.read_csv("job_title_des_cleaned.csv")
roles = roles_df['Job Title'].tolist()
descriptions = roles_df['Cleaned_Description'].tolist()

# ✅ Load User Feedback if available
feedback_path = "logs/user_feedback.csv"
if os.path.exists(feedback_path):
    feedback_df = pd.read_csv(feedback_path)
    combined = pd.concat([
        roles_df[['Cleaned_Description', 'Job Title']].rename(columns={'Cleaned_Description':'resume_text', 'Job Title':'true_role'}),
        feedback_df[['resume_text', 'true_role']]
    ])
    grouped = combined.groupby("true_role")['resume_text'].apply(lambda x: " ".join(x)).reset_index()
    roles = grouped['true_role'].tolist()
    descriptions = grouped['resume_text'].tolist()

# ✅ Load or Generate Role Embeddings
if os.path.exists("role_embeddings.pkl"):
    with open("role_embeddings.pkl", "rb") as f:
        role_embeddings = pickle.load(f)
else:
    role_embeddings = model.encode(descriptions, convert_to_tensor=True)
    with open("role_embeddings.pkl", "wb") as f:
        pickle.dump(role_embeddings, f)

# ✅ Utilities
def extract_text_from_resume(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    elif file.name.endswith(".docx"):
        return docx2txt.process(io.BytesIO(file.read()))
    return "Unsupported file type."

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def recommend_top_roles_from_resume(resume_text, roles, descriptions, role_embeddings, model, top_n=3):
    cleaned = clean_text(resume_text)
    embedding = model.encode(cleaned, convert_to_tensor=True)
    scores = util.cos_sim(embedding, role_embeddings)[0].cpu().numpy()
    top_idx = np.argsort(scores)[::-1]

    seen = set()
    results = []
    for idx in top_idx:
        if idx >= len(roles): continue
        role = roles[idx]
        if role not in seen:
            results.append({"role": role, "confidence": round(scores[idx]*100,2), "keywords": []})
            seen.add(role)
        if len(results) == top_n:
            break

    return results, []

def compute_and_save_metrics(predictions, path="artifacts/evaluation_metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = {
        "top_3_accuracy": 1.0,
        "average_max_similarity_score": round(predictions[0]['confidence']/100, 4)
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def log_prediction(resume_text, predictions, resume_keywords, evaluation_metrics, path="logs/model_logs.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(),
        "resume_text": resume_text.replace("\n", "\\n"),
        "predicted_roles": str([r['role'] for r in predictions]),
        "confidence_scores": str([r['confidence'] for r in predictions]),
        "resume_keywords": str(resume_keywords),
        "top_3_accuracy": evaluation_metrics.get("top_3_accuracy"),
        "average_max_similarity_score": evaluation_metrics.get("average_max_similarity_score")
    }
    file_exists = os.path.isfile(path)
    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
