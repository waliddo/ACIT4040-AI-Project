from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.losses import BinaryCrossentropy
from keras.layers import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class MetaModel:

    def __init__(self, n: int) -> None:
        self.n = n
        self.model = self.build_model()

    def predict(self, input: np.ndarray):
        self.model.predict(input)

    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(self.n)))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self, train_ds, test_ds, epochs: int):
        train_x, train_y = zip(*train_ds)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.model.compile(
            optimizer='adam',
            loss=BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'])

        history = self.model.fit(
            train_x,
            train_y,
            epochs=epochs,
            validation_split=0.2)

        return history

    def plot_accuracy(self, history):
        print(history.history.keys())
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1.1])
        plt.legend(loc='lower right')
        plt.show()

    def load(self, path: str):
        if path is None:
            return
        self.model = tf.keras.models.load_model(path)

    def save(self):
        self.model.save("models/meta_model.h5")
