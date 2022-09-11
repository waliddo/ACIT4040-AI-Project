import cv2
import torch
from math import log2

MODEL = 'models'
MODEL_NAME = 'faces_1'
MODEL_PATH = f'./{MODEL}/{MODEL_NAME}'

START_TRAIN_AT_IMG_SIZE = 64
DATASET = 'train_images'
CHECKPOINT_GEN = f'{MODEL_PATH}/generator.pth'
CHECKPOINT_DIS = f'{MODEL_PATH}/discriminator.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3

BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8, 4, 4]
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)

# BATCH_SIZES = [26, 26, 26, 12, 12, 6, 6, 4, 4] # to fit better with 156 images
# PROGRESSIVE_EPOCHS = [200, 200, 200, 200, 200, 200, 200, 200, 200] # for individual number of epochs

CHANNELS_IMG = 3
Z_DIM = 256  # 512 in original paper
IN_CHANNELS = 256  # 512 in original paper
DISCRIMINATOR_ITERATIONS = 1
LAMBDA_GP = 10
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 2