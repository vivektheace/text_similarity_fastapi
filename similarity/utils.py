import re
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

# model loader
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

# preprocessing data
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# computing similarity score
def predict_similarity(text1, text2):
    model = get_model()
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    score = cosine_similarity(embedding1, embedding2, dim=0)
    return round(score.item(), 4)

