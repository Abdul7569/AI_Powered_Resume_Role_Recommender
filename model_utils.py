import os
import io
import re
import json
import fitz, numpy as np, pandas as pd
import spacy
import pickle
from sentence_transformers import SentenceTransformer, util
import docx2txt

# Load model once
print("⏳ Loading model...")
model = SentenceTransformer('all-mpnet-base-v2')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load base job title dataset
df = pd.read_csv("job_title_des_cleaned.csv")
roles = df['Job Title'].tolist()
descriptions = df['Cleaned_Description'].tolist()


# Load and incorporate feedback into roles and descriptions
feedback_path = "logs/user_feedback.csv"
if os.path.exists(feedback_path):
    print("🔁 Using user feedback to improve model...")
    feedback_df = pd.read_csv(feedback_path)
    feedback_df.dropna(subset=["resume_text", "true_role"], inplace=True)
    feedback_df = feedback_df[feedback_df["true_role"].str.len() > 2]
    
    # Combine original and feedback data
    combined_df = pd.concat([
        df[['Cleaned_Description', 'Job Title']].rename(columns={
            'Cleaned_Description': 'resume_text',
            'Job Title': 'true_role'
        }),
        feedback_df[['resume_text', 'true_role']]
    ], ignore_index=True)

    # Group by role and generate one description per role
    grouped = combined_df.groupby("true_role")['resume_text'].apply(lambda x: " ".join(x)).reset_index()
    descriptions = grouped["resume_text"].tolist()
    
    descriptions = [str(desc).strip() for desc in descriptions if isinstance(desc, str) or isinstance(desc, int)]


    roles = grouped["true_role"].tolist()

# Only load or generate embeddings if needed
if os.path.exists("role_embeddings.pkl"):
    print("📦 Loading precomputed role embeddings...")
    with open("role_embeddings.pkl", "rb") as f:
        role_embeddings = pickle.load(f)
else:
    print("⚠️ No embeddings found. Generating and saving...")
    role_embeddings = model.encode(descriptions, convert_to_tensor=True)
    with open("role_embeddings.pkl", "wb") as f:
        pickle.dump(role_embeddings, f)
    print("✅ Embeddings saved.")

# ------------------- Utility Functions -------------------

def extract_text_from_resume(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(io.BytesIO(uploaded_file.read()))
    else:
        return "Unsupported file type."

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_keywords(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']]
    return list(set(skills))

def explain_match(resume_text, job_description):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    return list(resume_words & job_words)[:10]

def recommend_top_roles_from_resume(resume_text, roles, descriptions, role_embeddings, model, top_n=3):
    cleaned_resume = clean_text(resume_text)
    resume_embedding = model.encode(cleaned_resume, convert_to_tensor=True)
    similarity_scores = util.cos_sim(resume_embedding, role_embeddings)[0].cpu().numpy()
    sorted_indices = np.argsort(similarity_scores)[::-1]

    seen_titles = set()
    results = []
    for idx in sorted_indices:
        role = roles[idx]
        score = round(similarity_scores[idx] * 100, 2)
        explanation = explain_match(cleaned_resume, descriptions[idx])
        if role not in seen_titles:
            results.append({"role": role, "confidence": score, "keywords": explanation})
            seen_titles.add(role)
        if len(results) == top_n:
            break

    resume_keywords = extract_keywords(resume_text)
    return results, resume_keywords

def compute_and_save_metrics(predictions, save_path="artifacts/evaluation_metrics.json"):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    top_3_accuracy = 1.0  # placeholder (update based on ground truth if needed)
    avg_confidence = predictions[0]["confidence"] / 100  # convert from % if needed

    evaluation_metrics = {
        "top_3_accuracy": top_3_accuracy,
        "average_max_similarity_score": round(avg_confidence, 4)
    }

    with open(save_path, "w") as f:
        json.dump(evaluation_metrics, f, indent=4)

    return evaluation_metrics

def log_prediction(resume_text, predictions, resume_keywords, evaluation_metrics, log_path="logs/model_logs.csv"):
    import os, csv
    from datetime import datetime

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "resume_text": resume_text.replace("\n", "\\n"),
        "predicted_roles": str([item["role"] for item in predictions]),
        "confidence_scores": str([item["confidence"] for item in predictions]),
        "resume_keywords": str(resume_keywords),
        "top_3_accuracy": evaluation_metrics.get("top_3_accuracy"),
        "average_max_similarity_score": evaluation_metrics.get("average_max_similarity_score")
    }

    file_exists = os.path.isfile(log_path)
    fieldnames = list(row.keys())

    with open(log_path, mode="a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
