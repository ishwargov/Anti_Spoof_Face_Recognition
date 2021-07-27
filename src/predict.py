from joblib import load
from train import get_face,get_embedding
import numpy as np
import json
model = load('./data/ckpt_model.joblib')

with open('./data/train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

def predict_file(file_name):
    face = get_embedding(get_face(file_name))
    preds = model.predict_proba(face.reshape(1,128))
    preds2 = np.argmax(preds.reshape(-1))
    return(preds2,class_names[str(preds2)],preds[preds2])