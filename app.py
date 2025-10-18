from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import re

# Initialize FastAPI app
app = FastAPI(title="Text Similarity API")

# Define request model
class TextPair(BaseModel):
    text1: str
    text2: str

# Simple text preprocessing
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Compute similarity
def get_similarity(text1: str, text2: str) -> float:
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)
    emb1 = model.encode(clean_text1, convert_to_tensor=True)
    emb2 = model.encode(clean_text2, convert_to_tensor=True)
    sim_score = util.cos_sim(emb1, emb2).item()
    return round((sim_score + 1) / 2, 3)

# API route
@app.post("/similarity")
def similarity(payload: TextPair):
    score = get_similarity(payload.text1, payload.text2)
    return {"similarity_score": score}

# Root endpoint for testing
@app.get("/")
def home():
    return {"message": "Text Similarity API is running"}
