import cv2
import os
import color_features
import numpy as np
import tensorflow as tf


def load_imgs(dir: str) -> list:
    """
    Reads all images in a given directory.

    Args:
        dir (str): Path to a directory of images.

    Returns:
        list: All images.
    """
    imgs = []
    for file in os.listdir(dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in [".png", ".jpg", "jpeg"]:
            imgs.append(cv2.imread(os.path.join(dir, file)))
    return imgs


def load_data(fake_path: str, real_path: str) -> tuple[list, list]:
    """
    Load color data (color features) from images.

    Args:
        fake_path (str): Path to directory of fake images.
        real_path (str): Path to directory of real images.

    Returns:
        tuple[list, list]: (fake image features, real image features)
    """
    fake_imgs = load_imgs(fake_path)
    real_imgs = load_imgs(real_path)

    fake_features = [color_features.get_features(img) for img in fake_imgs]
    real_features = [color_features.get_features(img) for img in real_imgs]

    return (fake_features, real_features)


def load_train_test(fake_path: str,
                    real_path: str,
                    batch_size: int,
                    split: float) -> tuple[tuple[list, list]]:
    # Load data.
    fake, real = load_data(fake_path, real_path)
    fake = np.random.shuffle(np.array(fake))
    real = np.random.shuffle(np.array(real))
    # Split into train and test sets.
    train = []
    test = []
    train.append[fake[0:int(len(fake) * split)]]
    train.append[real[0:int(len(real) * split)]]
    train = tf.data.Dataset.from_tensor_slices(train) \
                           .shuffle(len(train)) \
                           .batch(batch_size)
    test.append[fake[int(len(fake) * split):]]
    test.append[real[int(len(real) * split):]]
    test = tf.data.Dataset.from_tensor_slices(test) \
                          .shuffle(len(test)) \
                          .batch(batch_size)
    return (train, test)


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(root, "..", "data")
    fake, real = load_data(dataset, dataset)
    print(np.shape(fake))
    print(np.shape(real))
