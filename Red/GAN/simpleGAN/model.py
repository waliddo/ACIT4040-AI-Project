'''
Inspired by:
https://www.tensorflow.org/tutorials/generative/dcgan
'''

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

# Loss function 
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

CROP_X, CROP_Y = 8, 8
WIDTH, HEIGHT, CHANNEL = 120, 120, 3
WIDTH_CROP, HEIGHT_CROP = WIDTH - (CROP_X * 2), HEIGHT - (CROP_Y * 2)
BUFFER_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 100

NOISE_DIM = 100
num_examples_to_generate = 16

S_TIME = str(int(time.time()))
MODEL = 'models'
MODEL_PATH = f'{MODEL}/{S_TIME}_new_faces'




def load_images():
    current_dir = os.getcwd()
    print(current_dir)
    images_dir = os.path.join(current_dir, 'train_images')
    images = []
    for image in os.listdir(images_dir):
        path = (os.path.join(images_dir, image))
        
        image = load_img(path, target_size=(WIDTH, HEIGHT))
        image = np.array(image)
        image = image[CROP_Y:HEIGHT-CROP_Y, CROP_X:WIDTH-CROP_X] # for cropping out borders

        # image = image.reshape(WIDTH, HEIGHT, CHANNEL)
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta = 0.1)
        # image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

        # image = (image - 127.5) / 127.5 # range [-1, 1] for grayscale
        image = image / 256 # range [0, 1] for colors
        
        image = tf.cast(image, tf.float32)
        images.append(image)
    
    return images


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(13*13*240, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((13, 13, 240)))
    assert model.output_shape == (None, 13, 13, 240)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(120, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 26, 26, 120)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(30, (10, 10), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 52, 52, 30)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 104, 104, 3)

    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(30, (10, 10), strides=(5, 5), padding='same',
                                     input_shape=[104, 104, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(120, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



generator = make_generator_model()
# generator.summary()
discriminator = make_discriminator_model()
# discriminator.summary()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = MODEL_PATH + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train():
    images = load_images()
    train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in train_dataset:
            train_step(image_batch)

        # save
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
            generate_and_save_images(generator, epoch + 1, seed) 

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    generate_and_save_images(generator, EPOCHS, seed)
    
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    i = 0
    for img in predictions:
        img_arry = np.array(img)
        image = (img_arry * 256).astype(int)
        plt.subplot(4, 4, i+1)
        plt.imshow(image)
        plt.axis('off')
        i += 1
    # plt.show()

    img_save_path = os.path.join(MODEL_PATH, 'epoch_images')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    plt.savefig(img_save_path + '/image_at_epoch_{:04d}.png'.format(epoch))

def test():
    checkpoint_dir = f'{MODEL_PATH}/training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    for _ in range(5):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        predictions = generator(noise, training=False)

        fig = plt.figure(figsize=(4, 4))

        for img in predictions:
            img_arry = np.array(img)
            image = (img_arry * 256).astype(int)
            plt.imshow(image)
            plt.axis('off')
        plt.show()



def main():
    train()
    # test()



if __name__ == "__main__":
    main()