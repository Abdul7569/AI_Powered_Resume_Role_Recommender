import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import streamlit as st  # Import streamlit if you intend to use it here
import json

# Initialize Firebase Admin SDK from Streamlit Secrets
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            cred_str = st.secrets["firebase_key"]
            print(f"Type of cred_str from secrets: {type(cred_str)}") # Debugging line
            print(f"Content of cred_str (first 50 chars): {cred_str[:50]}") # Debugging line
            cred_dict = json.loads(cred_str)
            print(f"Type of cred_dict after json.loads(): {type(cred_dict)}") # Debugging line
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/'
            })
            print("Firebase app initialized in firebase_utils.py")
        except KeyError:
            st.error("Firebase credentials not found in Streamlit secrets!")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Error decoding Firebase credentials from secrets: {e}")
            return None
        except Exception as e:
            st.error(f"Error initializing Firebase app in firebase_utils.py: {e}")
            return None
    return firebase_admin.get_app()

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

if __name__ == '__main__':
    # Example usage (for testing this file directly)
    if firebase_admin._apps:
        print("Firebase app is initialized. Running example uploads...")
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