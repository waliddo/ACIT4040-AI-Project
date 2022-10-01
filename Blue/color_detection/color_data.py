import cv2
import os
import color_features
import numpy as np


def load_imgs(dir: str, limit: int) -> list:
    """
    Reads all images in a given directory.

    Args:
        dir (str): Path to a directory of images.
        limit (int): Max amount of images to load.

    Returns:
        list: All images.
    """
    imgs = []
    for file in os.listdir(dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in [".png", ".jpg", "jpeg"]:
            imgs.append(cv2.imread(os.path.join(dir, file)))
            if len(imgs) >= limit:
                break
    return imgs


def write_imgs(imgs, dir):
    for index, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir, f"{index}.png"), img)


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
        features.append(color_features.get_features(img))
    return np.array(features)


def load_train_test(fake_path: str,
                    real_path: str) -> tuple[np.ndarray,
                                             np.ndarray,
                                             np.ndarray]:
    # Load and shuffle training data
    print("Loading train data..", end='')
    train = load_data(os.path.join(fake_path, "train"), 10)
    print("finished.")
    print("Shuffling train..", end="")
    np.random.shuffle(train)
    print("finished.")
    # Load test data and labels
    print("Loading test..", end="")
    test_fake = load_data(os.path.join(fake_path, "test"), 10)
    test_real = load_data(os.path.join(real_path, "test"), 1)
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
