from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import re
import os
import uvicorn

app = FastAPI(title="Text Similarity API")

class TextPair(BaseModel):
    text1: str
    text2: str

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_similarity(text1, text2):
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)

    emb1 = model.encode(clean_text1, convert_to_tensor=True)
    emb2 = model.encode(clean_text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    sim_score = (sim_score + 1) / 2 
    return round(sim_score, 3)

@app.post("/similarity")
def similarity(payload: TextPair):
    score = get_similarity(payload.text1, payload.text2)
    return {"similarity score": score}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
