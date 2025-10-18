#Import libraries
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re

# Basic text cleanup
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Load dataset
data = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Load transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Compute cosine similarity between two texts
def get_similarity(text1, text2):
    emb1 = model.encode(preprocess_text(text1), convert_to_tensor=True)
    emb2 = model.encode(preprocess_text(text2), convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round((score + 1) / 2, 3)  # Normalize to [0,1]

# Generate similarity scores for all pairs
data['similarity_score'] = [
    get_similarity(row['text1'], row['text2']) for _, row in data.iterrows()
]

# Save results
data.to_csv("results.csv", index=False)
