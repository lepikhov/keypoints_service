import io
import json
import os
import string
import time

#import numpy as np
#import tensorflow as tf
#from PIL import Image


def prepare_image(img_bytes):
    #print(img_bytes)
    pass

def predict_result(img):

    file = open("app/kp.json", "r")
    
    keypoints = json.loads(file.read())

    return keypoints
