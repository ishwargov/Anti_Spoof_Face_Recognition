from joblib import load
from train import get_face,get_embedding
import numpy as np
import json
import os
from PIL import Image
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
model = load('./data/ckpt_model.joblib')
spoof_model = load_model('./data/antispoof.h5')

with open('./data/train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

def predict_file(file_name):
    face = get_embedding(get_face(file_name))
    preds = model.predict_proba(face.reshape(1,128)).reshape(-1)
    preds2 = np.argmax(preds.reshape(-1))
    img = Image.open(file_name)
    img = img.resize((160,160))
    img = np.asarray(img)/255
    spoof_preds = spoof_model.predict(img.reshape(1,160,160,3)).reshape(-1)
    predictions = {}
    predictions['Class Name'] = class_names[str(preds2)]
    predictions['Confidence'] = preds[int(preds2)]
    predictions['Real'] = spoof_preds[1]
    predictions['Spoof'] = spoof_preds[0]
    return(predictions)