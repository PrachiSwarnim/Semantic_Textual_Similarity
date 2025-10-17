# similarity_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import re

# -------------------------------
# Minimal/light preprocessing
# -------------------------------
def preprocess_text(text):
    text = str(text).lower()                  # lowercase
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces & line breaks
    text = re.sub(r'[^\w\s]', '', text)       # remove punctuation
    text = text.strip()                        # trim spaces
    return text

# -------------------------------
# Load the model
# -------------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# -------------------------------
# Request body schema
# -------------------------------
class TextPair(BaseModel):
    text1: str
    text2: str

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Text Similarity API")

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/similarity")
def compute_similarity(payload: TextPair):
    clean_text1 = preprocess_text(payload.text1)
    clean_text2 = preprocess_text(payload.text2)

    emb1 = model.encode(clean_text1, convert_to_tensor=True)
    emb2 = model.encode(clean_text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    sim_score = (sim_score + 1) / 2  # scale to [0,1]
    return {"similarity score": round(sim_score, 3)}
