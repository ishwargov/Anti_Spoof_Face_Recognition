from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from joblib import dump

from app import num_of_classes,class_names,facenet,data


def get_face(filename):
  img = Image.open(filename)
  img = img.convert('RGB')
  img = np.asarray(img)
  face_detector = MTCNN()
  prediction = face_detector.detect_faces(img)
  x1, y1, w, h = prediction[0]['box'] 
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + w, y1 + h
  face = img[y1:y2,x1:x2]
  img = Image.fromarray(face)
  img = img.resize((160,160)) 
  face_array = np.asarray(img)
  face_array = face_array.astype('float32')
  mean, std = face_array.mean(), face_array.std()
  face_array = (face_array - mean) / std
  return face_array

def get_files(folder):
  faces = []
  for f in os.listdir(folder):
    faces.append(get_face(folder+'/'+f))
  return(faces)

def get_embedding(face):
  face = face.reshape(1,160,160,3)
  return (facenet.predict(face).reshape(-1,1))

def update_dataset(data,folder,class_name):
  global num_of_classes
  global class_names
  faces = get_files(folder)
  face_emb = [get_embedding(fc) for fc in faces]
  face_emb = np.array(face_emb)
  face_emb = face_emb.reshape(len(face_emb),-1)
  out = np.empty((len(face_emb),1))
  out.fill(num_of_classes)
  class_names[num_of_classes] = class_name
  num_of_classes += 1
  dat = np.concatenate([face_emb,out],axis=1)
  data = data.append(pd.DataFrame(dat, columns=[str(i) for i in np.arange(1,130)]), ignore_index=True)
  return data

def train(data):
    df = data.to_numpy()
    x = df[:,:128]
    y = df[:,128]
    model = SVC(kernel = 'linear',probability=True)
    model.fit(x,y)
    dump(model,'./data/ckpt_model.joblib')
    