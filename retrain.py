import pandas as pd
import pickle
from datetime import datetime
import os
from model_utils import model


# Load model


# Paths
log_path = "logs/model_logs.csv"
output_path = "role_embeddings.pkl"
role_data_path = "job_title_des.csv"

# Ensure logs exist
if not os.path.exists(log_path):
    print("❌ No logs found to retrain. Run some predictions first.")
    exit()

# Load prediction logs
df = pd.read_csv(log_path)
resume_texts = df["resume_text"].dropna().unique().tolist()

# Load original role descriptions
if not os.path.exists(role_data_path):
    print("❌ job_title_des.csv not found.")
    exit()

roles_df = pd.read_csv(role_data_path)
combined_descriptions = roles_df['Cleaned_Description'].dropna().tolist()

# Combine for retraining
combined_inputs = resume_texts + combined_descriptions
print(f"✅ Total retraining inputs: {len(combined_inputs)}")

# Generate updated embeddings
updated_embeddings = model.encode(combined_inputs, convert_to_tensor=False)

# Save updated embeddings
with open(output_path, "wb") as f:
    pickle.dump(updated_embeddings, f)

print(f"✅ New role embeddings saved to: {output_path}")
print(f"🕒 Timestamp: {datetime.now().isoformat()}")
