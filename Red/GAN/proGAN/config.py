import torch
from math import log2

MODEL = 'models'
MODEL_NAME = 'celeba_hq_hillary'
NAME_EXTENSION = ''
MODEL_PATH = f'./{MODEL}/{MODEL_NAME}'

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'faceswaps'
CHECKPOINT_GEN = f'{MODEL_PATH}/512_generator.pth'
CHECKPOINT_DIS = f'{MODEL_PATH}/512_discriminator.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
SAVE_IMAGES = False
SAVE_IAMGES_LAYER = 7

BATCH_SIZES = [32, 32, 32, 32, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)

# BATCH_SIZES = [26, 26, 26, 12, 12, 6, 6, 4, 4] # to fit better with 156 images
# PROGRESSIVE_EPOCHS = [10, 20, 30, 50, 70, 100, 100, 100] # for individual number of epochs

LEARNING_RATE = 1e-3
CHANNELS_IMG = 3
Z_DIM = 256  # 512 in original paper
IN_CHANNELS = 256  # 512 in original paper
DISCRIMINATOR_ITERATIONS = 1
LAMBDA_GP = 10
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 2
