import os
from keras.preprocessing.image import array_to_img, img_to_array,load_img
from keras.utils import np_utils
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


class image_cut:
    def __init__(self):
        self.dir = ["ayaneru","hanazawakana","hayamisaori","kayanoai","taketatsuayane","tomatsuharuka","yuukiaoi"]
        self.url = "./voice_actor/face/"
        self.list_x = []
        self.list_y = []
        self.image_size = 224

    def file_insert(self):
        
        for index,dire in enumerate(self.dir):
            face_file = glob.glob(self.url + dire + "/*.jpg")
            count=0
            for face in face_file:
                if count==50:
                    break

                with Image.open(face) as f:
                    image = f.convert("RGB")
                    image = image.resize((self.image_size, self.image_size))
                    data = np.asarray(image)
                    self.list_x.append(data)
                    self.list_y.append(index)
                count+=1
            

        self.list_x = np.array(self.list_x)
        self.list_y = np.array(self.list_y)
        
        self.list_x = self.list_x.astype('float32')
        self.list_x = self.list_x / 255.0
        # 正解ラベルの形式を変換
        self.list_y = np_utils.to_categorical(self.list_y, len(self.dir))
    
        X_train, X_test, y_train, y_test = train_test_split(self.list_x, self.list_y, test_size=0.20)
        return X_train, X_test, y_train, y_test        
        


    def process(self):
        pass

if __name__ == "__main__":
    image = image_cut()
    image.file_insert()



