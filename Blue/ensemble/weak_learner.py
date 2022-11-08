import numpy as np
from abc import ABC, abstractmethod


class WeakLearner(ABC):

    @abstractmethod
    def predict(image: np.ndarray) -> np.ndarray:
        """
        Predict whether an image is real or fake.

        Args:
            img (np.ndarray): Image(s)

        Returns:
            np.ndarray: Prediction where 0 means fake and 1 means real.
        """
        raise NotImplementedError()
