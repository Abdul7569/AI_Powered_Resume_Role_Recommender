# sync_firebase_logs.py

import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import json
from datetime import datetime

# --- Initialize Firebase ---
if not firebase_admin._apps:
    try:
        # Assuming you have FIREBASE_KEY in your GitHub secrets
        cred_json = os.getenv("FIREBASE_KEY")
        if cred_json:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/'
            })
        else:
            raise ValueError("No FIREBASE_KEY found in environment.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        exit(1)

# --- Pull Data from Firebase ---
def download_from_firebase(node_path, local_csv_path, columns):
    try:
        ref = db.reference(node_path)
        data = ref.get()

        if data:
            records = list(data.values())
            df = pd.DataFrame(records)

            # Ensure correct columns
            for col in columns:
                if col not in df.columns:
                    df[col] = ""

            df = df[columns]
            os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
            df.to_csv(local_csv_path, index=False)
            print(f"✅ Downloaded and saved {node_path} to {local_csv_path}")
        else:
            print(f"⚠️ No data found at {node_path}")

    except Exception as e:
        print(f"❌ Error downloading {node_path}: {e}")

# --- Paths and Columns ---
user_feedback_csv = "logs/user_feedback.csv"
model_logs_csv = "logs/model_logs.csv"

user_feedback_columns = ["timestamp", "resume_text", "predicted_role", "true_role"]
model_logs_columns = ["timestamp", "resume_text", "predicted_roles", "confidence_scores", "resume_keywords", "top_3_accuracy", "average_max_similarity_score"]

# --- Download ---
download_from_firebase("user_feedback", user_feedback_csv, user_feedback_columns)
download_from_firebase("model_logs", model_logs_csv, model_logs_columns)

print("\n🎯 Firebase sync complete.")
