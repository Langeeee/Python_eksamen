import os
import cv2
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join

class machine_trainer_test:

    def __init__(self) -> None:
        pass
    
    def make_and_save_mnist_model(self):

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)


        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))


        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs = 3)

        loss, accuracy = model.evaluate(evaluation)

        print(loss)
        print(accuracy)

        model.save('handwritten.model')



    def get_keras_model(self):
        return tf.keras.models.load_model('handwritten.model')


    def check_images(self, model, image_number):
        mypath = '/home/jovyan/Python_eksamen/Images/CutImages/' + str(image_number)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
       # print(onlyfiles)
        for x in range(5):
           
            try:
                 
                img = cv2.imread(mypath + '/' + onlyfiles[x])[:,:,0]
                img2 = np.invert(np.array([img]))
                prediction = model.predict(img2)

                plt.imshow(img, cmap=plt.cm.binary)
               
                plt.title(f"this digit is probably a {np.argmax(prediction)}")
               # fig.suptitle(f"this digit is probably a {np.argmax(prediction)}", fontsize=16)
                
                plt.figure()
                
               # print(f"this digit is probably a {np.argmax(prediction)}")
                
            except: 
                
                traceback.print_exc()
            