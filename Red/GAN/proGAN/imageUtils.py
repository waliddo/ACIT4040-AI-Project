import os
import scipy.misc
import tensorflow as tf
from PIL import Image


def save_images_from_event(fn, tag, output_dir='tf_images'):
    '''
    Code for extracting images from an tf event file, does not work
    
    code from:
    https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary
    '''
    
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1


def resize_folder(source_folder, out_folder, target_x=512, target_y=512):
    if os.path.isdir(source_folder):
        
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        count = 0
        for file in os.listdir(source_folder):
            image = Image.open(f'{source_folder}/{file}')
            image = image.resize((target_x, target_y), Image.ANTIALIAS)
            image.save('{}/image_{:04d}.png'.format(out_folder, count))
            count += 1

resize_folder('hilary', 'iteration6', 4114, 516)
# resize_folder('real_fake_images/hillary/faceswap', 'real_fake_images/hillary/out')
# save_images_from_event('logs/events.out.tfevents.1664548985.g001.11420.0', 'faces_2 Fake')