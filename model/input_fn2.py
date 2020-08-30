import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
import pdb
import random


def parser(record):
    keys_to_features = {"image": tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.string),
                        "mask": tf.FixedLenFeature([], tf.string)}
    # images = tf.image.decode_image(image_content, channels=self.channels[0])
    # masks = tf.image.decode_image(mask_content, channels=self.channels[1])
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed["label"], tf.uint8)
    label = tf.cast(label, tf.float32)
    mask = tf.decode_raw(parsed["mask"], tf.uint8)
    mask = tf.cast(mask, tf.float32)
    image = tf.reshape(image, shape=[112, 112, 3])
    label = tf.reshape(label, shape=[112, 112, 3])
    mask = tf.reshape(mask, shape=[112, 112, 3])
    image = preprocess_input(image)
    label = preprocess_input(label)
    return image, label, mask


def _corrupt_brightness(image, mask):
    """
    Radnomly applies a random brightness change.
    """
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask


def _corrupt_contrast(image, mask):
    """
    Randomly applies a random contrast change.
    """
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _corrupt_saturation(image, mask):
    """
    Randomly applies a random saturation change.
    """
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _crop_random(image, mask, seed):
    """
    Randomly crops image and mask in accord.
    """
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    shape = tf.cast(tf.shape(image), tf.float32)
    h = tf.cast(shape[0] * crop_percent, tf.int32)
    w = tf.cast(shape[1] * crop_percent, tf.int32)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [h, w, channels[0]], seed=seed), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [h, w, channels[1]], seed=seed), lambda: tf.identity(mask))

    return image, mask


def _flip_left_right(image, mask, seed):
    """
    Randomly flips image and mask left or right in accord.
    """
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask


def input_fn(is_training, filenames, batch_size=32, n_epoch=1, n_threads=16, augment=True, n_buffer=8, seed=None):
    if seed is None:
        seed = random.randint(0, 10000)
    if is_training:
        dataset = tf.data.TFRecordDataset(filenames=filenames,
                                          num_parallel_reads=n_threads)
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(4096*4, n_epoch))
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(parser, batch_size,
                                               num_parallel_batches=n_threads))
        # If augmentation is to be applied
        if augment:
            dataset = dataset.map(_corrupt_brightness,
                                  num_parallel_calls=n_threads).prefetch(n_buffer)

            dataset = dataset.map(_corrupt_contrast,
                                  num_parallel_calls=n_threads).prefetch(n_buffer)

            dataset = dataset.map(_corrupt_saturation,
                                  num_parallel_calls=n_threads).prefetch(n_buffer)

            dataset = dataset.map(_crop_random,
                                  num_parallel_calls=n_threads).prefetch(n_buffer)

            dataset = dataset.map(_flip_left_right,
                                  num_parallel_calls=n_threads).prefetch(n_buffer)

        dataset = dataset.prefetch(buffer_size=2)
    else:
        dataset = tf.data.TFRecordDataset(filenames=filenames,
                                          num_parallel_reads=n_threads)
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(4096*4, n_epoch))
        # dataset = dataset.apply(
        #     tf.data.experimental.shuffle_and_repeat(1024, n_epoch))
        # dataset = dataset.map(parser, num_parallel_calls=8)
        # dataset = dataset.batch(batch_size)
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(parser, batch_size,
                                               num_parallel_batches=n_threads))
        dataset = dataset.prefetch(buffer_size=n_buffer)
    iterator = dataset.make_one_shot_iterator()
    # init_op = iterator.make_initializer(dataset)
    # images, labels, masks = iterator.get_next()
    return iterator


def train_inputs_fn(b_size, n_epoch):
    return input_fn(True, filenames=["../data/train0to9maskRandom.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 215009


def valid_inputs_fn(b_size, n_epoch):
    return input_fn(False, filenames=["../data/valid10to12mask.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 65276


def test_inputs_fn(b_size, n_epoch):
    return input_fn(False, filenames=["../data/test50.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 2269


def runtime_inputs_fn(b_size):
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    masks = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, 224, 224, 3])
    inputs = {'images': images, 'masks': masks, 'labels': labels}
    return inputs
