import os
import cv2
import numpy as np
import matplotlib as plt
import tensorflow as tf

class machine_trainer:

    def __init__(self) -> None:
        pass

    def make_dataset(self):
        batch_size = 32
        img_height = 28
        img_width = 28

        train_ds = tf.keras.utils.image_dataset_from_directory(
          "/home/jovyan/Python_eksamen/Images/CutImages/",
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
          "/home/jovyan/Python_eksamen/Images/CutImages/",
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )
        class_names = train_ds.class_names
        print(class_names)
        return train_ds, val_ds

    def train_model(self, train_ds, val_ds):
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
          epochs=51
        )

        train_loss, train_accuracy = model.evaluate(train_ds)
        val_loss, val_accuracy = model.evaluate(val_ds)


        print("training loss: " + str(train_loss))
        print("val_loss: " + str(val_loss))
        print("training accuracy: " + str(train_accuracy))
        print("val accuracy: " + str(val_accuracy))

        model.save('verifiedCaptcha.model')
    #https://www.tensorflow.org/tutorials/load_data/images

