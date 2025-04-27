import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# Load your Firebase Admin SDK key
cred = credentials.Certificate("firebase_key.json")

# Connect to Firebase Realtime Database
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/'
})


def upload_model_log(resume_text, predicted_roles, confidence_scores, resume_keywords):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "resume_text": resume_text,
        "predicted_roles": predicted_roles,
        "confidence_scores": confidence_scores,
        "resume_keywords": resume_keywords
    }
    ref = db.reference("model_logs")
    ref.push(log_data)
    print("✅ Model log uploaded.")

def upload_user_feedback(resume_text, predicted_role, true_role):
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "resume_text": resume_text,
        "predicted_role": predicted_role,
        "true_role": true_role
    }
    ref = db.reference("user_feedback")
    ref.push(feedback_data)
    print("✅ User feedback uploaded.")
