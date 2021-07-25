from joblib import load
from train import get_face,get_embedding
from app import class_names
import numpy as np
model = load('ckpt_model.joblib')

def predict(file_name):
    face = get_embedding(get_face(file_name))
    preds = model.predict_proba(face)
    preds2 = np.argmax(preds.reshape(-1))
    print(preds)
    print(preds2,class_names[preds2])