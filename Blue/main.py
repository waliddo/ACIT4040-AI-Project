from ensemble.ensemble_model import EnsembleModel
import os
from data.data_loader import load_data


if __name__ == '__main__':
    ensemble = EnsembleModel()
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "data", "saved_images")
    real_path = ""
    test_imgs, test_labels = load_data(real_path, fake_path, 50, rgb=True)

    ensemble.evaluate(test_imgs, test_labels)
