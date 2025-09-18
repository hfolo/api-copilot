from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Leer dataset (asegurate de subir dataset_reviews.xlsx junto con el código)
df = pd.read_excel("dataset_reviews.xlsx", header=1)
texts = df["texto"].tolist()
labels = df["etiqueta"].tolist()

# Generar embeddings del dataset
embeddings = model.encode(texts, convert_to_tensor=True)

# Inicializar FastAPI
app = FastAPI(title="Clasificador por Embeddings")

class Query(BaseModel):
    texto: str

@app.get("/")
def root():
    return {"mensaje": "API funcionando en Azure 🚀"}

@app.post("/classify")
def classify(query: Query):
    emb_new = model.encode(query.texto, convert_to_tensor=True)
    cos_scores = util.cos_sim(emb_new, embeddings)[0].cpu().numpy()
    top_k = 3
    idx_sorted = cos_scores.argsort()[::-1][:top_k]

    resultados = []
    for i in idx_sorted:
        resultados.append({
            "texto_similar": texts[i],
            "etiqueta": labels[i],
            "score": float(cos_scores[i])
        })

    etiquetas_top = [labels[i] for i in idx_sorted]
    etiqueta_pred = max(set(etiquetas_top), key=etiquetas_top.count)

    return {
        "input": query.texto,
        "prediccion": etiqueta_pred,
        "vecinos": resultados
    }