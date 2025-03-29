from fastapi import FastAPI
from pydantic import BaseModel
from similarity.utils import predict_similarity

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Semantic Similarity API is running 🚀"}

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def get_similarity(request: SimilarityRequest):
    score = predict_similarity(request.text1, request.text2)
    return {"similarity score": score}

