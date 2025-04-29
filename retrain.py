import pandas as pd
import pickle
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials, db
from sentence_transformers import SentenceTransformer
import json

# ✅ Initialize Firebase (same logic as you have in firebase_utils.py)
if not firebase_admin._apps:
    try:
        if "firebase_key" in os.environ:
            cred_json = os.environ["firebase_key"]
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
        else:
            import streamlit as st
            cred = credentials.Certificate(dict(st.secrets["firebase_key"]))

        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/'
        })
        print("✅ Firebase initialized inside retrain.py")
    except Exception as e:
        print(f"⚠️ Error initializing Firebase in retrain.py: {e}")
        exit()

# ✅ Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Paths
output_path = "role_embeddings.pkl"
role_data_path = "job_title_des_cleaned.csv"

# ✅ Load base job title dataset
data_frames = []

if os.path.exists(role_data_path):
    roles_df = pd.read_csv(role_data_path)
    base_df = roles_df[['Cleaned_Description', 'Job Title']].rename(columns={
        'Cleaned_Description': 'resume_text',
        'Job Title': 'true_role'
    })
    data_frames.append(base_df)
else:
    print("❌ job_title_des_cleaned.csv not found.")
    exit()

# ✅ Pull model_logs from Firebase
try:
    logs_ref = db.reference('model_logs')
    logs_data = logs_ref.get()

    if logs_data:
        logs_df = pd.DataFrame.from_dict(logs_data, orient='index')
        logs_df = logs_df[['resume_text']].dropna().copy()
        logs_df['true_role'] = 'unlabeled'
        data_frames.append(logs_df)
    print(f"✅ Pulled {len(logs_df)} records from model_logs.")
except Exception as e:
    print(f"⚠️ Error fetching model_logs: {e}")

# ✅ Pull user_feedback from Firebase
try:
    feedback_ref = db.reference('user_feedback')
    feedback_data = feedback_ref.get()

    if feedback_data:
        feedback_df = pd.DataFrame.from_dict(feedback_data, orient='index')
        feedback_df = feedback_df[['resume_text', 'true_role']].dropna()
        feedback_df = feedback_df[feedback_df['true_role'].str.len() > 2]
        data_frames.append(feedback_df)
    print(f"✅ Pulled {len(feedback_df)} records from user_feedback.")
except Exception as e:
    print(f"⚠️ Error fetching user_feedback: {e}")

if not data_frames:
    print("❌ No valid data found for retraining.")
    exit()

# ✅ Combine all sources
combined_df = pd.concat(data_frames, ignore_index=True)

# ✅ Group by role (aggregate multiple resumes per role)
grouped = combined_df.groupby("true_role")['resume_text'].apply(lambda x: " ".join(x)).reset_index()
descriptions = grouped["resume_text"].tolist()
roles = grouped["true_role"].tolist()

print(f"✅ Total retraining roles: {len(roles)}")

# ✅ Generate updated embeddings
updated_embeddings = model.encode(descriptions, convert_to_tensor=False)

# ✅ Save updated embeddings
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(updated_embeddings, f)

print(f"✅ New role embeddings saved to: {output_path}")
print(f"🕒 Timestamp: {datetime.now().isoformat()}")
