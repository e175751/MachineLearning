import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import sys
from Image_cut import image_cut

def main(op,ep):
    image = image_cut()
    X_train, X_test, y_train, y_test = image.file_insert()
    save_path = "./save/model"
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    print("study start")
    model = Model(X_train,op)
    epoch = int(ep)
    
    model.fit(X_train, y_train, epochs=epoch, verbose=1)

    model.save(save_path + "cnn_" + op + ep + ".h5")
    print(model.evaluate(X_test, y_test))
    print("finish")

def Model(X_train,op):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

#    model.add(Conv2D(128, (3, 3), padding='same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(128, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.5))

    #model.add(Conv2D(256, (3, 3), padding='same'))
    #model.add(Activation('relu'))
    #model.add(Conv2D(256, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7))
    model.add(Activation('softmax'))    


    # コンパイル
    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])    
    return model


if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])

#-----------------------------------------------------------------------
#Warning (initialization): An error occurred while loading ‘/Users/e17
