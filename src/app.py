import json
from train import update_dataset
from keras.models import load_model
import pandas as pd
import os
from flask import Flask
app = Flask(__name__)

with open('train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

facenet = load_model('./data/facenet_keras.h5')
data = pd.read_csv('./data/train.csv')

@app.route("/")
def home():
    return "<p>Face Recognition</p>"
