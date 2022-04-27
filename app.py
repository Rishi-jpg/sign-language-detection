from crypt import methods
from email import message
import imp
import importlib
from ntpath import join
from re import template
import re
from cv2 import norm
from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import cv2
import skfuzzy
from skfuzzy.membership import gaussmf
from prediction_service.prediction import predict


webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


def brightness(img):
    if len(img.shape) == 3:
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        return np.average(img)


def adjust_brightness(img, thresh=100):
    # Checking Brigtness Level
    c = 0
    if brightness(img) < thresh:
        # setting a value for adjusting brightness level
        value = int(thresh-brightness(img))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Recurssion
        if gaussmf(np.asarray([brightness(img)]), thresh, 50) < 0.95 and c <= 100:
            c += 1
            adjust_brightness(img)

        return img

    else:
        value = int(brightness(img)-thresh)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        if gaussmf(np.asarray([brightness(img)]), thresh, 50) < 0.95:
            c += 1
            adjust_brightness(img)

        return img


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/data", methods=["GET", "POST"])

def data():
    if request.method == "POST":
        if request.files:
            try:
                os.mkdir('data/')
            except:
                pass
            image = request.files['image']
            image.save('data/'+'test'+'.png')

            img_path = 'data/test.png'
            
            messege = predict(img_path)
            return render_template('index.html', messege=messege)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
