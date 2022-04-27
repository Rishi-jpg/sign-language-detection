import yaml
import os
import json
import joblib
import numpy as np
import keras
import cv2
import pandas as pd
import tensorflow

print(tensorflow.__version__)

IMAGE_CHANNEL = 3
def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath)
    new_array = np.reshape(img_array, (IMG_SIZE, IMG_SIZE, IMAGE_CHANNEL))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNEL)

def predict(img_path):
    base = "prediction_service"
    model_dir_path = os.path.join(base,"model")
    model_path = os.path.join(model_dir_path, "my_h5_model.h5")
    
    model = keras.models.load_model(model_path)

    Name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
    N = list(range(len(Name)))
    # normal_mapping = dict(zip(Name,N))
    reverse_mapping = dict(zip(N,Name))
   
    pred = model.predict([prepare(img_path)])
    PRED = []
    for item in pred:
        value = np.argmax(item)
        PRED += [value]

    # print(pd.Series(PRED).value_counts())

    predict = reverse_mapping[PRED[0]]
    # print(predict)
    
    return predict

