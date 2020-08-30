import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
import pdb


def parser(record):
    keys_to_features = {"image": tf.io.FixedLenFeature([], tf.string),
                        # "label": tf.io.FixedLenFeature([], tf.string),
                        "mask": tf.io.FixedLenFeature([], tf.string),
                        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'mcase': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    # label = tf.decode_raw(parsed["label"], tf.uint8)
    # label = tf.cast(label, tf.float32)
    mask = tf.decode_raw(parsed["mask"], tf.uint8)
    mask = tf.cast(mask, tf.float32)

    height = tf.cast(parsed["height"], tf.int64)
    width = tf.cast(parsed["width"], tf.int64)
    mtype = tf.cast(parsed["mcase"], tf.int64)

    image = tf.reshape(image, shape=[height, width, 3])
    mask = tf.reshape(mask, shape=[height, width, 3])
    mtype = K.equal(mtype, 1)
    return image, mask, mtype


def _brightness(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(f_flip, lambda: tf.image.random_brightness(
        image, 0.1), lambda: tf.identity(image))
    return image, mask, mtype


def _contrast(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(f_flip, lambda: tf.image.random_contrast(
        image, 0, 1), lambda: tf.identity(image))
    return image, mask, mtype


def _hue(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(f_flip, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask, mtype


def _saturation(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(f_flip, lambda: tf.image.random_saturation(
        image, 0, 1), lambda: tf.identity(image))
    return image, mask, mtype


def _bw(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(f_flip, lambda: tf.image.rgb_to_grayscale(
        image), lambda: tf.identity(image))
    return image, mask, mtype


def _vgg_preproc(image, mask, mtype):
    image = tf.clip_by_value(image, 0, 255)
    image = image[:, :, ::-1]-[103.939, 116.779, 123.68]
    return image, mask, mtype


def _crop(image, mask, mtype):
    image = tf.image.random_crop(image, [224, 224, 3])
    mask = tf.image.random_crop(mask, [224, 224, 3])
    return image, mask, mtype


def _hflip(image, mask, mtype):
    f_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(
        f_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.image.random_flip_left_right(mask)
    return image, mask, mtype


def input_fn(filenames, batch_size=32, n_epoch=1, n_threads=4, f_augment=True, is_valid=False):
    dataset = tf.data.TFRecordDataset(filenames=filenames,
                                      num_parallel_reads=n_threads)
    dataset = dataset.repeat()
    # if is_valid:
    #     print('is_vald')
    #     dataset = dataset.shuffle(1024)
    # else:
    dataset = dataset.shuffle(4096)
    dataset = dataset.map(parser, n_threads)
    if f_augment:
        dataset = dataset.map(_brightness, n_threads)
        dataset = dataset.map(_contrast, n_threads)
        dataset = dataset.map(_hue, n_threads)
        dataset = dataset.map(_saturation, n_threads)
        dataset = dataset.map(_bw, n_threads)
        dataset = dataset.map(_hflip, n_threads)
        dataset = dataset.map(_vgg_preproc, n_threads)
        dataset = dataset.map(_crop, n_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=3)
    return tf.compat.v1.data.make_one_shot_iterator(dataset)


def train_inputs_fn(b_size, n_epoch):
    return input_fn(filenames=["../data/train0to9maskRandom.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 215009


def valid_inputs_fn(b_size, n_epoch):
    return input_fn(filenames=["../data/valid10to12maskRandom.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch, is_valid=True), 65276


def test_inputs_fn(b_size, n_epoch):
    return input_fn(False, filenames=["../data/test50.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 2269


def runtime_inputs_fn(b_size):
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    masks = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, 224, 224, 3])
    inputs = {'images': images, 'masks': masks, 'labels': labels}
    return inputs
