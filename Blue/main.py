from ensemble.ensemble_model import EnsembleModel
import os
import numpy as np
from data.data_loader import load_train_test


if __name__ == '__main__':
    ensemble = EnsembleModel()
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "data", "celebahq", "fake")
    real_path = os.path.join(root, "data", "celebahq", "real")
    train_imgs, train_labels, test_imgs, test_labels = \
        load_train_test(real_path, 100, fake_path, 100, rgb=True)
    print(np.shape(train_imgs))
    print(np.shape(train_labels))
    print(np.shape(test_imgs))
    print(np.shape(test_labels))

    ensemble.train(train_imgs, train_labels, test_imgs, test_labels, 25)
