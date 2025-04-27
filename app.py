# ✅ Streamlit page config
st.set_page_config(page_title="AI Job Recommender", page_icon="🤖", layout="centered")
import asyncio
import streamlit as st
import pandas as pd
import os
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from model_utils import (
    extract_text_from_resume,
    recommend_top_roles_from_resume,
    roles,
    descriptions,
    role_embeddings,
    log_prediction,
    clean_text,
    compute_and_save_metrics
)
from evaluate import evaluate_model
# from firebase_utils import upload_user_feedback # Commented out to address potential import issues
from firebase_admin import credentials, initialize_app

# 🛡️ FIX: Ensure there is a running event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load model using Streamlit caching
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    return model

model = load_model()

# ✅ Initialize Firebase Admin SDK from Streamlit Secrets
if not initialize_app._apps:
    try:
        cred_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(cred_dict)
        initialize_app(cred, {
            'databaseURL': 'https://resume-role-recommender-default-rtdb.firebaseio.com/' # Replace with your actual URL if different
        })
        print("Firebase app initialized using Streamlit secrets.")
    except KeyError:
        st.error("Firebase credentials not found in Streamlit secrets!")
    except Exception as e:
        st.error(f"Error initializing Firebase app from secrets: {e}")

# ✅ Feedback log path
LOG_PATH = "logs/user_feedback.csv"
expected_columns = ["timestamp", "resume_text", "predicted_role", "true_role"]

# ✅ Ensure feedback file structure
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

if not os.path.exists(LOG_PATH):
    st.info("📁 Creating new feedback file with correct headers...")
    df = pd.DataFrame(columns=expected_columns)
    df.to_csv(LOG_PATH, index=False)
else:
    df = pd.read_csv(LOG_PATH, on_bad_lines='skip')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_columns]
    df.dropna(subset=["resume_text", "true_role"], inplace=True)
    df.to_csv(LOG_PATH, index=False)



# ✅ App Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📄 AI-Powered Job Role Recommender</h1>", unsafe_allow_html=True)
st.markdown("Upload your **resume** in `.pdf` or `.docx` format and get top matching job roles using AI 🧠")

# ✅ File uploader
uploaded_file = st.file_uploader("📎 Upload Resume", type=["pdf", "docx"])

# ✅ Main logic
if uploaded_file:
    with st.spinner("⚙️ Extracting and analyzing your resume..."):
        resume_text = extract_text_from_resume(uploaded_file)
        results, skills = recommend_top_roles_from_resume(resume_text, roles, descriptions, role_embeddings, model)
        evaluation_metrics = compute_and_save_metrics(results)
        log_prediction(resume_text, results, skills, evaluation_metrics)

    st.success("✅ Analysis Complete!")

    st.markdown("## 🎯 Top Job Role Matches")

    for i, res in enumerate(results):
        confidence = res['confidence']
        color = "green" if confidence > 80 else "orange" if confidence > 60 else "red"

        with st.container():
            st.markdown(f"""
                <div style='padding:10px; border: 1px solid #ddd; border-radius:10px; margin-bottom: 10px; background-color: #f9f9f9;'>
                    <h4>🔹 {i+1}. <span style='color: #2E86C1;'>{res['role']}</span></h4>
                    <div style='margin-bottom: 6px;'>Confidence Score: <strong style='color:{color};'>{confidence}%</strong></div>
                    <div style='margin-bottom: 6px;'>📌 <b>Top Keywords:</b> {', '.join(res['keywords']) if res['keywords'] else 'N/A'}</div>
                    <progress value='{confidence}' max='100' style='width: 100%; height: 12px;'></progress>
                </div>
            """, unsafe_allow_html=True)

    if skills:
        with st.expander("🛠️ View Extracted Skills from Resume"):
            st.code(", ".join(skills), language="markdown")

# ✅ Evaluation Section
accuracy, similarity_scores_list = evaluate_model()
st.title("🧪 Model Evaluation Dashboard")

st.metric(label="Top-3 Accuracy", value=f"{accuracy * 100:.2f}%")
st.bar_chart(similarity_scores_list)

# ✅ Feedback Section - Smarter Version
st.title("📝 Resume Role Recommender Feedback")
st.markdown("""
### ✅ If our prediction didn't match your real role, help us improve!
Fill in the form below to log your real role. This feedback is used to retrain the model automatically.
""")

with st.form("feedback_form"):
    feedback_resume_text = st.text_area("Paste your resume or job summary", height=200)
    predicted_role = st.text_input("What did our app predict for you?")
    correct_role = st.text_input("What is your actual role?")
    submit = st.form_submit_button("Submit Feedback")

if submit:
    if not feedback_resume_text.strip():
        st.error("⚠️ Please paste your resume text.")
    elif not predicted_role.strip():
        st.error("⚠️ Please enter what role our app predicted.")
    elif not correct_role.strip():
        st.error("⚠️ Please enter your actual (correct) role.")
    else:
        # Ensure firebase_admin app is initialized before using it
        if initialize_app._apps:
            try:
                from firebase_utils import upload_user_feedback # Import here to avoid issues if Firebase isn't initialized
                upload_user_feedback(
                    resume_text=feedback_resume_text,
                    predicted_role=predicted_role.strip(),
                    true_role=correct_role.strip()
                )
                st.success("🎉 Thanks! Your feedback has been recorded and uploaded!")
            except ImportError as e:
                st.error(f"⚠️ Error: Could not import 'upload_user_feedback'. Ensure 'firebase_utils.py' exists and is correctly implemented: {e}")
            except Exception as e:
                st.error(f"⚠️ Error uploading feedback to Firebase: {e}")
        else:
            st.warning("⚠️ Firebase Admin SDK not initialized. Feedback not uploaded.")