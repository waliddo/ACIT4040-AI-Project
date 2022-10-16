from resnet50 import ResNet50
from inception import Inception
from color_detection.color_classifier import ColorClassifier
from ensemble.weak_learner import WeakLearner
from typing import List
import numpy as np
from ensemble.meta_model import MetaModel


class EnsembleModel():

    def __init__(self) -> None:
        self.weak_learners: List[WeakLearner] = []

        color_clf = ColorClassifier(bgr=False)
        color_clf.load("models/color_svm.joblib")
        self.weak_learners.append(color_clf)
        self.weak_learners.append(ResNet50())
        self.weak_learners.append(Inception())

        self.meta_model = MetaModel(3)
        self.meta_model.load(None)

    def predict(self, imgs: np.ndarray):
        return self.meta_model.predict(self._get_learner_predictions(imgs))

    def _get_learner_predictions(self, images):
        predictions = None
        for weak_learner in self.weak_learners:
            _ = weak_learner.predict(images)
            _ = np.reshape(_, (len(images)))
            if predictions is None:
                predictions = _
            else:
                predictions = np.dstack([predictions, _])
        return predictions

    def train(self,
              train_imgs, train_labels,
              test_imgs, test_labels,
              epochs: int):
        train_preds = self._get_learner_predictions(train_imgs)
        test_preds = self._get_learner_predictions(test_imgs)
        train_preds = np.squeeze(train_preds)
        test_preds = np.squeeze(test_preds)

        history = self.meta_model.train(
            zip(train_preds, train_labels),
            zip(test_preds, test_labels),
            epochs)
        self.meta_model.plot_accuracy(history)
        self.meta_model.save()
