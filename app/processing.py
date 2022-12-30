import io
import json
import os
import string
import time

import numpy as np
from app.calculator import calcs
from app.config import W1, W2
from PIL import Image


def predict_result(img_bytes):

    #file = open("app/kp.json", "r")

    #keypoints = json.loads(file.read())
    
    img = calcs[0].prepare_image(img_bytes)
    kp1 = calcs[0].predict(img) 

    img = calcs[1].prepare_image(img_bytes)
    kp2 = calcs[1].predict(img)    

    size = min(len(kp1), len(kp2))

    keypoints=[]

    for i in range(size):
        keypoints.append([(kp1[i][0]*W1 + kp2[i][0]*W2)/100, (kp1[i][1]*W1 + kp2[i][1]*W2)/100])

    return keypoints
