import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import util, SentenceTransformer
from model_utils import clean_text, roles, role_embeddings
from difflib import get_close_matches
from config.load_config import load_config
import os, json

# Load updated evaluation dataset
df = pd.read_csv("Sample_Evaluation_Data.csv")
print(df.head())
print(f"✅ Loaded {len(df)} evaluation samples.")

def evaluate_model(model_path="Sample_Evaluation_Data.csv", top_k=3):
    model = SentenceTransformer('all-mpnet-base-v2')
    df = pd.read_csv(model_path)

    correct = 0
    similarity_scores_list = []

    for i, row in df.iterrows():
        cleaned = clean_text(row['resume_text'])
        embedding = model.encode(cleaned, convert_to_tensor=True)
        similarity_scores = util.cos_sim(embedding, role_embeddings)[0].cpu().numpy()
        similarity_scores_list.append(similarity_scores.max())

        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
       
        top_roles = [roles[idx] for idx in top_indices if idx < len(roles)]


        true_role = row['true_role'].strip().lower()
        predicted_roles_cleaned = [r.strip().lower() for r in top_roles]

        if true_role in predicted_roles_cleaned:
            print("✅ MATCHED!")
            correct += 1
        else:
            match = get_close_matches(true_role, predicted_roles_cleaned, n=1, cutoff=0.8)
            if match:
                print(f"🔶 Fuzzy Match Found: {match[0]}")
                correct += 1
            else:
                print("❌ Not Matched.")

    accuracy = correct / len(df)
    return accuracy, similarity_scores_list

# Run evaluation
accuracy, similarity_scores_list = evaluate_model()
print("🎯 Top-3 Accuracy:", round(accuracy, 4))

# Save metrics
config = load_config()
metrics_output_path = config["evaluation"]["metrics_output_path"]

metrics = {
    "top_3_accuracy": round(float(accuracy), 4),
    "average_max_similarity_score": round(float(np.mean(similarity_scores_list)), 4)
}

os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
with open(metrics_output_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Metrics saved to: {metrics_output_path}")

# Plot
plt.figure(figsize=(8, 5))
plt.hist(similarity_scores_list, bins=10, color="skyblue", edgecolor="black")
plt.title("Distribution of Max Similarity Scores")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
