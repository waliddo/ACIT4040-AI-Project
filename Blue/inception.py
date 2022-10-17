from ensemble.weak_learner import WeakLearner
import numpy as np
import tensorflow as tf


class Inception(WeakLearner):

    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('models/inception.hdf5')

    def predict(self, img: np.ndarray) -> np.ndarray:
        img = tf.image.resize(img, (224, 224))
        return self.model(img)
