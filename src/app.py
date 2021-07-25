import json
from keras.models import load_model
from flask_ngrok import run_with_ngrok
import pandas as pd
import os
from flask import Flask
app = Flask(__name__)
run_with_ngrok(app) 

with open('./data/train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

facenet = load_model('./data/facenet_keras.h5')
data = pd.read_csv('./data/train.csv')

from train import update_dataset
@app.route("/")
def home():
    return ("<p>Face Recognition</p>")

if __name__ == '__main__':
    app.run()