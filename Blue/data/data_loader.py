import cv2
import os
import numpy as np


def load_imgs(dir: str, limit: int, rgb=False) -> list:
    """
    Reads all images in a given directory.

    Args:
        dir (str): Path to a directory of images.
        limit (int): Max amount of images to load.
        rgb (bool): Return the image converted to RGB instead of default BGR.

    Returns:
        list: All images.
    """
    imgs = []
    for file in os.listdir(dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in [".png", ".jpg", "jpeg"]:
            img = cv2.imread(os.path.join(dir, file))
            if rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            if len(imgs) >= limit:
                break
    return imgs


def load_data(real_dir: str, fake_dir: str, n: int, rgb: bool):
    real = load_imgs(real_dir, int(n / 2), rgb=rgb)
    fake = load_imgs(fake_dir, int(n / 2), rgb=rgb)
    real_fake = np.concatenate([real, fake])

    fake_labels = np.zeros(shape=(len(real)))
    real_labels = np.ones(shape=(len(fake)))
    labels = np.concatenate([real_labels, fake_labels])
    labels = labels.astype(np.uint8)
    idx = np.random.permutation(len(real_fake))
    real_fake = real_fake[idx]
    labels = labels[idx]

    return (real_fake, labels)


def load_train_test(real_dir: str, train_n: int,
                    fake_dir: str, test_n: int, rgb: bool):
    train, train_labels = load_data(
        os.path.join(real_dir, "train"),
        os.path.join(fake_dir, "train"),
        train_n,
        rgb=rgb)

    test, test_labels = load_data(
        os.path.join(real_dir, "test"),
        os.path.join(fake_dir, "test"),
        test_n,
        rgb=rgb)

    return (train, train_labels, test, test_labels)


def write_imgs(imgs, dir):
    for index, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir, f"{index}.png"), img)


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(root, "celebahq", "fake")
    real_path = os.path.join(root, "celebahq", "real")
    train, train_labels, test, test_labels = \
        load_train_test(real_path, 10, fake_path, 10, 1)
    print(np.shape(train))
    print(np.shape(train_labels))
