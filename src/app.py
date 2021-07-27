import json
from keras.models import load_model
from flask_ngrok import run_with_ngrok
from flask import *
import pandas as pd
import os
from predict import predict_file
app = Flask(__name__)
run_with_ngrok(app) 

with open('./data/train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

facenet = load_model('./data/facenet_keras.h5')
data = pd.read_csv('./data/train.csv')

app.config['UPLOAD_FOLDER'] = './static/uploads'

from train import update_dataset
@app.route("/")
def home():
    return (render_template('home.html'))

@app.route("/upload")
def upload():
    return( render_template('upload.html'))

@app.route('/predict', methods = ['POST'])  
def predict():  
    if request.method == 'POST':  
        f = request.files['file'] 
        path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename) 
        f.save(path)  
        preds = predict_file(path)
        return render_template("predict.html", filename=path, name = preds[1])  

if __name__ == '__main__':
    app.run()