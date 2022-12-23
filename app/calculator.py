import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import MODEL_PATH


class Calculator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224,224)

    def predict(self, img):
        #raise ValueError
        kp = self.model.predict(img)
        kp = np.reshape(kp, (kp.size//2,2), order='C')   
        w, h = self.get_img_size()     
        kp = [[x[0]*w, h-x[1]*h] for x in kp]
        return kp

    def set_img_size(self, size):
        self.img_size = size

    def get_img_size(self):
        return self.img_size   

        



calc = Calculator(MODEL_PATH)
