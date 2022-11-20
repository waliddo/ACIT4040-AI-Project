from resnet50 import ResNet50
from inception import Inception
from color_detection.color_classifier import ColorClassifier
from ensemble.weak_learner import WeakLearner
from typing import List
import numpy as np
from ensemble.meta_model import MetaModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class EnsembleModel():

    def __init__(self) -> None:
        self.weak_learners: List[WeakLearner] = []

        color_clf = ColorClassifier(bgr=False)
        color_clf.load("models/color_svm.joblib")
        self.weak_learners.append(color_clf)
        self.weak_learners.append(ResNet50())
        self.weak_learners.append(Inception())

        self.meta_model = MetaModel(3)
        self.meta_model.load("models/meta_model.h5")

    def predict(self, imgs: np.ndarray):
        preds = np.squeeze(self._get_learner_predictions(imgs))
        return self.meta_model.predict(preds)

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

    def evaluate(self, test_imgs, test_labels):
        predictions = np.squeeze(np.rint(self.predict(test_imgs)).astype(np.uint8))
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        print(f"Accuracy: {accuracy}")

        # conf_matrix = confusion_matrix(test_labels, predictions)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        # for i in range(conf_matrix.shape[0]):
        #     for j in range(conf_matrix.shape[1]):
        #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        # plt.xticks([0, 1], ['Deepfake', 'Real'])
        # plt.yticks([0, 1], ['Deepfake', 'Real'])
        # plt.xlabel('Predictions', fontsize=18)
        # plt.ylabel('Actuals', fontsize=18)
        # plt.title('Confusion Matrix', fontsize=18)
        # plt.show()
