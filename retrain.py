import pandas as pd
import pickle
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Paths
log_path = "logs/model_logs.csv"
feedback_path = "logs/user_feedback.csv"
output_path = "role_embeddings.pkl"
role_data_path = "job_title_des_cleaned.csv"

# Load datasets
data_frames = []

if os.path.exists(role_data_path):
    roles_df = pd.read_csv(role_data_path)
    base_df = roles_df[['Cleaned_Description', 'Job Title']].rename(columns={
        'Cleaned_Description': 'resume_text',
        'Job Title': 'true_role'
    })
    data_frames.append(base_df)
else:
    print("❌ job_title_des_Cleaned.csv not found.")

if os.path.exists(log_path):
    logs_df = pd.read_csv(log_path)
    logs_df = logs_df[['resume_text']].dropna().copy()
    logs_df['true_role'] = 'unlabeled'
    data_frames.append(logs_df)

if os.path.exists(feedback_path):
    feedback_df = pd.read_csv(feedback_path)
    feedback_df = feedback_df[['resume_text', 'true_role']].dropna()
    feedback_df = feedback_df[feedback_df['true_role'].str.len() > 2]
    data_frames.append(feedback_df)

if not data_frames:
    print("❌ No valid data found for retraining.")
    exit()

# Combine all sources
combined_df = pd.concat(data_frames, ignore_index=True)

# Group by role
grouped = combined_df.groupby("true_role")['resume_text'].apply(lambda x: " ".join(x)).reset_index()
descriptions = grouped["resume_text"].tolist()
roles = grouped["true_role"].tolist()

print(f"✅ Total retraining roles: {len(roles)}")

# Generate updated embeddings
updated_embeddings = model.encode(descriptions, convert_to_tensor=False)

# Save updated embeddings
with open(output_path, "wb") as f:
    pickle.dump(updated_embeddings, f)

print(f"✅ New role embeddings saved to: {output_path}")
print(f"🕒 Timestamp: {datetime.now().isoformat()}")
