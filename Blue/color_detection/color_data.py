import os
import color_detection.color_features as color
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import load_imgs


def load_data(path: str, n: int) -> list:
    """
    Load color data (color features) from images.

    Args:
        path (str): Path to directory of images.
        n (int): Amount to load.

    Returns:
        np.ndarray: Color features for each image.
    """
    imgs = load_imgs(path, n)
    features = []
    for img in imgs:
        features.append(color.get_features(img))
    return np.array(features)


def load_train_test(fake_path: str,
                    real_path: str) -> tuple[np.ndarray,
                                             np.ndarray,
                                             np.ndarray]:
    """
    Load train and test set for color detection.

    Args:
        fake_path (str): Path to directory with fake images.
        real_path (str): Path to directory with real images.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (train, test, test labels)
    """
    # Load and shuffle training data
    print("Loading train data..", end='')
    train = load_data(os.path.join(real_path, "train"), 10000)
    print("finished.")
    print("Shuffling train..", end="")
    np.random.shuffle(train)
    print("finished.")
    # Load test data and labels
    print("Loading test..", end="")
    test_fake = load_data(os.path.join(fake_path, "test"), 2000)
    test_real = load_data(os.path.join(real_path, "test"), 2000)
    print("finished.")
    fake_labels = np.zeros(shape=(len(test_fake)))
    real_labels = np.ones(shape=(len(test_real)))
    test_labels = np.concatenate([fake_labels, real_labels])
    test = np.concatenate([test_fake, test_real])
    # Shuffle test data and labels
    idx = np.random.permutation(len(test))
    test = test[idx]
    test_labels = test_labels[idx]

    return (train, test, test_labels)


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "..", "data", "celebahq", "fake", "test")
    real_path = os.path.join(root, "..", "data", "celebahq", "real", "test")
    fake_imgs = load_imgs(fake_path, 1000)
    real_imgs = load_imgs(real_path, 1000)
    fake_features = [color.get_features(img) for img in fake_imgs]
    real_features = [color.get_features(img) for img in real_imgs]
    fake_avrg = np.mean(fake_features, axis=0)
    real_avrg = np.mean(real_features, axis=0)
    diff = np.abs(fake_avrg - real_avrg)
    x = range(1, 589)
    ax1 = plt.subplot(1, 3, 1)
    plt.plot(x, fake_avrg, color='red', alpha=0.75)
    plt.title("Mean Color Features - Fake Images")
    plt.ylim(top=0.7, bottom=0.0)
    ax2 = plt.subplot(1, 3, 2)
    plt.plot(x, real_avrg, color='blue', alpha=0.75)
    plt.title("Mean Color Features - Real Images")
    plt.ylim(top=0.7, bottom=0.0)
    ax3 = plt.subplot(1, 3, 3)
    plt.plot(x, diff, color='orange', alpha=0.75)
    plt.title("Difference")
    plt.ylim(top=0.7, bottom=0.0)
    plt.show()
