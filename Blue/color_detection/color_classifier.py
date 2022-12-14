from sklearn.linear_model import SGDOneClassSVM
import color_detection.color_features as color
import numpy as np
import joblib
import color_detection.color_data
import os
import cv2
from ensemble.weak_learner import WeakLearner


class ColorClassifier(WeakLearner):
    """
    Classifies images as real or fake based on its color features.
    """

    def __init__(self, bgr: bool) -> None:
        self.is_bgr = bgr
        self.classifier = SGDOneClassSVM(nu=0.85)

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Predict whether an image is real or fake based on its color features.

        Args:
            img (np.ndarray): Image(s) in BGR form (default for cv2).
            rgb (bool): If the image(s) is in RGB format.

        Returns:
            np.ndarray: Prediction where 0 means fake and 1 means real.
        """
        if len(np.shape(img)) == 3:
            return self.predict_once(img)
        elif len(np.shape(img)) == 4:
            predictions = []
            for image in img:
                predictions.append(self.predict_once(image)[0])
            return np.array(predictions)
        else:
            raise ValueError(f"Invalid image shape: {np.shape(img)}")

    def predict_once(self, img: np.ndarray):
        if not self.is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        features = color.get_features(img)
        features = np.reshape(features, (1, len(features)))
        prediction = self.classifier.predict(features)
        return np.clip(prediction, 0, 1)

    def train(self, data):
        """
        Train the model using color features from training data.

        Args:
            data (np.ndarray): Color features for training images.
        """
        self.classifier = self.classifier.fit(data)

    def evaluate(self, test: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Evaluate the accuracy of the model.

        Args:
            test (np.ndarray): Color features for test images.
            test_labels (np.ndarray): Labels (0 = Fake, 1 = Real)

        Returns:
            float: The accuracy in range [0, 1].
        """
        predictions = np.clip(np.array(self.classifier.predict(test)), 0, 1)
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        return accuracy

    def save(self, path: str):
        joblib.dump(self.classifier, path)

    def load(self, path: str):
        self.classifier = joblib.load(path)

    def optimize(self, train, test, test_labels):
        """
        Trains and optimizes a model by testing various hyperparameters
        through grid search.

        Args:
            train (np.ndarray): Training images.
            test (np.ndarray): Test images.
            test_labels (np.ndarray): Labels where fake=0 and real=1.
        """
        nu_grid = [0.05, 0.1, 0.15, 0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        best_clf = None
        best_acc = 0.0
        best_nu = 0.0
        for nu in nu_grid:
            self.classifier = SGDOneClassSVM(nu=nu)
            self.train(train)
            acc = self.evaluate(test, test_labels)
            if acc > best_acc:
                best_clf = self.classifier
                best_acc = acc
                best_nu = nu
                print("New best classifier | " +
                      f"nu = {best_nu} | " +
                      f"accuracy = {best_acc}")
        joblib.dump(best_clf,
                    os.path.join(root, "..", "models", "optimized.joblib"))


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "..", "data", "celebahq", "fake")
    real_path = os.path.join(root, "..", "data", "celebahq", "real")

    clf = ColorClassifier(bgr=True)

    # Train the model
    # train, test, test_labels = \
    #     color_data.load_train_test(fake_path, real_path)
    # clf.optimize(train, test, test_labels)

    # Test model
    # clf.load(os.path.join(root, "..", "models", "color_svm.joblib"))
    # data = color_data.load_data(os.path.join(root, "..", "data", "fakes"))
    # acc = clf.evaluate(data, np.zeros(shape=(len(data))))
    # print(f"Accuracy = {acc}")
