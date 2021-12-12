import os
import cv2
import numpy as np
import matplotlib as plt
import tensorflow as tf

class machine_trainer_test:

    def __init__(self) -> None:
        pass
    
    def make_and_safe_dataset(self):

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)


        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Flatten(input_shape=(28.28)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))


        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']

        model.fit(x_train, y_train, epochs = 3)


        model.save('handwritten.model')

    def get_model(self):
        return tf.keras.models.load_model('handwritten.model')

    def evaluate_model(self):

        model = get_model()
        loss, accuracy = model.evaluate(x_test, y_test)

        print(loss)
        print(accuracy)

    


    def check_images(self, model):
        image_number = 1
        while os.path.isfile(f"/Images/CutImages/" + {image_number}.jpeg):
            try:
                img = cv2.imread(f"/Images/CutImages/" + {image_number}.jpeg)[:,:,0]
                img = np.invert(np.array([img]))
                prediction = model.predict(jpeg)
                print(f"this digit is probably a {np.argmax(prediction)}")
                plt.imshow(jpeg[0], cmap=plt.cm.binary)
                plt.show()
            except: 
                print("Error!")
            finally:
                image_number += 1