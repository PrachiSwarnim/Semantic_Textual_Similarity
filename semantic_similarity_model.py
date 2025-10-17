from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re

# -------------------------------
# Minimal/light preprocessing
# -------------------------------
def preprocess_text(text):
    text = str(text).lower()                   # lowercase
    text = re.sub(r'\s+', ' ', text)           # collapse multiple spaces & line breaks
    text = re.sub(r'[^\w\s]', '', text)        # remove punctuation
    text = text.strip()                         # trim spaces
    return text

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Load semantic similarity model (MPNet)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# -------------------------------
# Compute similarity function
# -------------------------------
def get_similarity(text1, text2):
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)

    emb1 = model.encode(clean_text1, convert_to_tensor=True)
    emb2 = model.encode(clean_text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    
    # Rescale from [-1,1] to [0,1]
    sim_score = (sim_score + 1) / 2
    return round(sim_score, 3)

# -------------------------------
# Apply model on dataset
# -------------------------------
similarity_scores = []

for idx, row in data.iterrows():
    text1 = row['text1']
    text2 = row['text2']
    score = get_similarity(text1, text2)
    similarity_scores.append(score)
    
    # Print row index + score + text snippets
    print(f"Row {idx}")
    print(f"Text1 snippet: {text1[:500]}")
    print(f"Text2 snippet: {text2[:500]}")
    print(f"Similarity Score: {score}")
    print("-" * 80)

data['similarity_score'] = similarity_scores

# Save results
data.to_csv("results.csv", index=False)
print("✅ Similarity scores saved to results.csv (0-1 scale).")
