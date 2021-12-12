import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import traceback
import cv2
import numpy as np
from matplotlib import pyplot as plt 

class machine_trainer:

    def __init__(self) -> None:
        pass

    def make_dataset(self, dir_path):
        batch_size = 32
        img_height = 28
        img_width = 28

        train_ds = tf.keras.utils.image_dataset_from_directory(
          dir_path,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
          dir_path,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )
        class_names = train_ds.class_names
        print(class_names)
        return train_ds, val_ds

    def train_and_save_model(self, train_ds, val_ds, model_name, epoch_amt):
        num_classes = 10
        model = tf.keras.Sequential([
          tf.keras.layers.Rescaling(1./255),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(num_classes)
        ])
        model.compile(
          optimizer='adam',
          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])
        model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epoch_amt
        )
        train_loss, train_accuracy = model.evaluate(train_ds)
        val_loss, val_accuracy = model.evaluate(val_ds)

        print("training loss: " + str(train_loss))
        print("val_loss: " + str(val_loss))
        print("training accuracy: " + str(train_accuracy))
        print("val accuracy: " + str(val_accuracy))

        model.save(model_name + '.model')


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

        train_loss, train_accuracy = model.evaluate(x_train, y_train)
        val_loss, val_accuracy = model.evaluate(x_test, y_test)


        print("training loss: " + str(train_loss))
        print("val_loss: " + str(val_loss))
        print("training accuracy: " + str(train_accuracy))
        print("val accuracy: " + str(val_accuracy))

        model.save('handwritten.model')

    def get_mnist_ds(self):
      mnist = tf.keras.datasets.mnist
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      return (x_train, y_train), (x_test, y_test)

    def get_keras_model(self):
        return tf.keras.models.load_model('handwritten.model')

    def get_verified_captcha_model(self):
        return tf.keras.models.load_model('verifiedCaptcha.model')
    
    def get_scraped_captcha_model(self):
        return tf.keras.models.load_model('scrapedCaptcha.model')
        
    def check_images(self, model, image_number):
        mypath = '/home/jovyan/Python_eksamen/Images/CutImages/' + str(image_number)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for x in range(5):
            try:
                img = cv2.imread(mypath + '/' + onlyfiles[x])[:,:,0]
                img2 = np.invert(np.array([img]))
                prediction = model.predict(img2)

                plt.imshow(img, cmap=plt.cm.binary)
                plt.title(f"this digit is probably a {np.argmax(prediction)}")
                plt.figure()
            except:  
                traceback.print_exc()
    #https://www.tensorflow.org/tutorials/load_data/images

