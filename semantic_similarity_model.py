from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
def preprocess_text(text):
    text = str(text).lower()                   # lowercase
    text = re.sub(r'\s+', ' ', text)           # collapse multiple spaces & line breaks
    text = re.sub(r'[^\w\s]', '', text)        # remove punctuation
    text = text.strip()                         # trim spaces
    return text

data = pd.read_csv("DataNeuron_Text_Similarity.csv")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_similarity(text1, text2):
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)

    emb1 = model.encode(clean_text1, convert_to_tensor=True)
    emb2 = model.encode(clean_text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    
    sim_score = (sim_score + 1) / 2
    return round(sim_score, 3)

similarity_scores = []

for idx, row in data.iterrows():
    text1 = row['text1']
    text2 = row['text2']
    score = get_similarity(text1, text2)
    similarity_scores.append(score)

data['similarity_score'] = similarity_scores

data.to_csv("results.csv", index=False)
