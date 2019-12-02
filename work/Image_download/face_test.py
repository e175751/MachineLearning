import os
from keras.preprocessing.image import array_to_img, img_to_array,load_img
from keras.models import load_model
from PIL import Image
import numpy as np

def main():
    x=[]
    url = "./voice_actor/face/kayanoai/img_99.jpg87-69.jpg"
    #url = "/Users/e175751/Desktop/tomatu.png"
    dire = ["ayaneru","hanazawakana","hayamisaori","kayanoai","taketatsuayane","tomatsuharuka","yuukiaoi"]
    image = Image.open(url)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    data = np.asarray(image)
    x.append(data)

    x = np.array(x)
    x = x.astype('float32')
    x = x / 255.0

    model = load_model("save/modelcnn_Adamax18.h5", compile=False)
    pred = model.predict(x, batch_size=1, verbose=0)
    score = np.max(pred)
    #判別結果の配列から最も高いところを抜きだし、そのクラス名をpred_labelへ
    pred_label = dire[np.argmax(pred[0])]
    #表示
    print('name:',pred_label)
    print('score:',score)

if __name__ == "__main__":
    main()

