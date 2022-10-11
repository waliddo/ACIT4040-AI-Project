from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.losses import BinaryCrossentropy
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
        model.add(Input(shape=(None, self.n)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self, train_ds, test_ds, epochs: int):
        self.model.compile(
            optimizer='adam',
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['accuracy', 'val_accuracy'])

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            validation_steps=2)

        return history

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def load(self, path: str):
        if path is None:
            return
        self.model = tf.keras.models.load_model(path)

    def save(self):
        self.model.save("./meta_model")
