import io
import json
import os
import string
import time

import numpy as np
from PIL import Image

from app.calculator import calc


def prepare_image(img_bytes):

    img = Image.open(io.BytesIO(img_bytes))
    calc.set_img_size(img.size)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):

    #file = open("app/kp.json", "r")

    #keypoints = json.loads(file.read())
    
    keypoints = calc.predict(img) 

    return keypoints
