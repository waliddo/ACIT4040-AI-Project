from ensemble.weak_learner import WeakLearner
import numpy as np
import tensorflow as tf


class ResNet50(WeakLearner):

    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('models/resnet.h5')

    def predict(self, img: np.ndarray) -> np.ndarray:
        img = tf.image.resize(img, (224, 224))
        return self.model(img)
