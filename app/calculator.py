import io

import numpy as np
import tensorflow as tf
from app.config import MODEL1_PATH, MODEL2_PATH
from app.models.model import BlazePose
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



class Calculator:
    def __init__(self, type, model_path, model=None, model_input_size=(224,224)):
        if type=='sequential' or type=='functional':            
            self.model = tf.keras.models.load_model(model_path)
        elif type=='subclassing':
            self.model = model  
            self.model.build((None, model_input_size[0], model_input_size[1],3))  
            self.model.load_weights(model_path)  
        else:
            raise ValueError('type must be sequential, functional or subclassing')          
        self.img_size = (224,224)
        self.model_input_size = model_input_size
        self.type = type

    def prepare_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes))
        self.set_img_size(img.size)
        img = img.resize(self.model_input_size)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        return img        

    def predict(self, img):
        w, h = self.get_img_size() 
        if self.type=='sequential' or self.type=='functional':
            kp = self.model.predict(img)
            kp = np.reshape(kp, (kp.size//2,2), order='C')  
            kp = [[x[0]*w, h-x[1]*h] for x in kp]
        elif self.type=='subclassing':
            kpt = self.model(tf.convert_to_tensor(img, dtype=tf.float32))   
            kp=[] 
            for idx in range(kpt.shape[1]):
                x = float(w*kpt[0][idx][0]/self.model_input_size[0])
                y = float(h-h*kpt[0][idx][1]/self.model_input_size[1])
                kp.append([x,y])
        else:
            raise ValueError('type must be sequential, functional or subclassing') 
        #print('type=',self.type, kp)
        return kp

    def set_img_size(self, size):
        self.img_size = size

    def get_img_size(self):
        return self.img_size   

        



calcs = [
    Calculator(type='sequential', model_path=MODEL1_PATH),
    Calculator(type='subclassing', model_path=MODEL2_PATH, model=BlazePose(), model_input_size=(256,256)),
]    
