from sklearn.linear_model import SGDOneClassSVM
import color_features
import numpy as np
import joblib
import color_data
import os


class ColorClassifier:
    """
    Classifies images as real or fake based on its color features.
    """

    def __init__(self) -> None:
        self.classifier = SGDOneClassSVM(nu=0.85)

    def predict(self, image):
        """
        Predict whether an image is real or fake based on its color features.

        Args:
            image (np.ndarray): Image in BGR form (default for cv2).

        Returns:
            np.ndarray: Prediction where 0 means fake and 1 means real.
        """
        features = color_features.get_features(image)
        prediction = self.classifier.predict(features)
        return np.clip(prediction, 0, 1)

    def predict_all(self, images):
        """
        Create a prediction for all given images.

        Args:
            images (np.ndarray): Array of images in BGR format.

        Returns:
            np.ndarray: Prediction where 0 means fake and 1 means real.
        """
        features = [color_features.get_features(image) for image in images]
        return np.clip(self.classifier.predict(features), 0, 1)

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
        print(predictions)
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        print(f"Accuracy: {round(accuracy, 3)}")
        return accuracy

    def save(self, path: str):
        joblib.dump(self.classifier, path)

    def load(self, path: str):
        self.classifier = joblib.load(path)


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "..", "data", "celebahq", "fake")
    real_path = os.path.join(root, "..", "data", "celebahq", "real")

    clf = ColorClassifier()
    train, test, test_labels = color_data.load_train_test(fake_path, real_path)
    clf.load(os.path.join(root, "..", "models", "color_svm_84.joblib"))
    clf.evaluate(test, test_labels)
    # clf.train(train)
    # clf.evaluate(test, test_labels)
    # clf.save(os.path.join(root, "..", "models", "color_svm.joblib"))
