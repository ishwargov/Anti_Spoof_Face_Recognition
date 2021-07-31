import json
from flask_ngrok import run_with_ngrok
from flask import *
import pandas as pd
import os
from predict import predict_file
from train import update_dataset,train_new
app = Flask(__name__)
run_with_ngrok(app) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

with open('./data/train_data.json') as td:
    train_data = json.load(td)
num_of_classes = train_data['num_of_classes']
class_names = train_data['class_names']

#facenet = load_model('./data/facenet_keras.h5')

app.config['UPLOAD_FOLDER'] = './static/uploads'

@app.route("/")
def home():
    return (render_template('home.html'))

@app.route("/upload")
def upload():
    return( render_template('upload.html'))

@app.route("/upload_data")
def upload_data():
    return( render_template('upload_data.html'))

@app.route('/predict', methods = ['POST'])  
def predict():  
    if request.method == 'POST':  
        f = request.files['file'] 
        path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename) 
        f.save(path)  
        preds = predict_file(path)
        return render_template("predict.html", filename=path, pred = preds)  

@app.route('/train', methods = ['POST'])  
def train(): 
    if request.method == 'POST':  
        f = request.files['file'] 
        path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename) 
        f.save(path)
        dataset = pd.read_csv('./data/train.csv')
        os.system('unzip '+path+' -d '+path[:-4])
        num_of_files = len(os.listdir(path[:-4]))
        dataset = update_dataset(dataset,path[:-4],f.filename[:-4])
        dataset.to_csv('./data/train.csv')
        os.system('rm -rf '+path)
        os.system('rm -rf '+path[:-4])
        train_new(dataset)
        return(f'<p>Successfully Trained {num_of_files} images , added 1 class. </p>')

if __name__ == '__main__':
    app.run()
