import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os

# ✅ Smart initialize Firebase: works for Streamlit, GitHub Actions, local
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            if "firebase_key" in os.environ:
                # Running in GitHub Actions or local (from .env or secrets)
                import json
                cred_json = os.environ["firebase_key"]
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
            else:
                # Running inside Streamlit
                import streamlit as st
                cred = credentials.Certificate(dict(st.secrets["firebase_key"]))
            
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/'
            })
            print("✅ Firebase app initialized successfully!")
        
        except Exception as e:
            print(f"⚠️ Error initializing Firebase app: {e}")

# ✅ Upload model log to Firebase
def upload_model_log(resume_text, predicted_roles, confidence_scores, resume_keywords):
    try:
        ref = db.reference("model_logs")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "resume_text": resume_text,
            "predicted_roles": predicted_roles,
            "confidence_scores": confidence_scores,
            "resume_keywords": resume_keywords
        }
        ref.push(log_data)
        print("✅ Model log uploaded to Firebase.")
    except Exception as e:
        print(f"⚠️ Error uploading model log to Firebase: {e}")

# ✅ Upload user feedback to Firebase
def upload_user_feedback(resume_text, predicted_role, true_role):
    try:
        ref = db.reference("user_feedback")
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "resume_text": resume_text,
            "predicted_role": predicted_role,
            "true_role": true_role
        }
        ref.push(feedback_data)
        print("✅ User feedback uploaded to Firebase.")
    except Exception as e:
        print(f"⚠️ Error uploading user feedback to Firebase: {e}")

# ✅ Optional: Example testing if running this file standalone
if __name__ == "__main__":
    initialize_firebase()
    if firebase_admin._apps:
        upload_model_log(
            "Example resume text",
            ["Software Engineer", "Data Analyst"],
            [90, 85],
            ["Python", "SQL", "Machine Learning"]
        )
        upload_user_feedback(
            "Another example resume",
            "Machine Learning Engineer",
            "AI Researcher"
        )
    else:
        print("⚠️ Firebase app not initialized. Cannot run example uploads.")
