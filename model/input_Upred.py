import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
import pdb

def _vgg_preproc(image, mask, mtype):
    image = tf.clip_by_value(image, 0, 255)
    image = image[:, :, ::-1]-[103.939, 116.779, 123.68]
    return image, mask, mtype

def _crop(image, mask, mtype):
    # print(image.shape)
    image = tf.image.central_crop(image, .36)
    # image = tf.image.central_crop(image, .18)
    # mask = tf.image.central_crop(mask, .16)
    return image, mask, mtype

def _resize(image, mask, mtype):
    # print(image.shape)
    image = tf.image.resize(image, [84, 84])
    # image = tf.image.resize(image, [42, 42])
    # mask = tf.image.central_crop(mask, .16)
    return image, mask, mtype

def parser(record):
    keys_to_features = {"image": tf.io.FixedLenFeature([], tf.string),
                        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'label': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        'im_ind': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    height = tf.cast(parsed["height"], tf.int64)
    width = tf.cast(parsed["width"], tf.int64)
    im_ind = tf.cast(parsed["im_ind"], tf.int64)
    label = tf.cast(parsed["label"], tf.float32)
    # label = tf.sign(tf.cast(parsed["label"], tf.float32))
    image = tf.reshape(image, shape=[height, width, 3])
    masks = K.constant(np.ones((42, 42, 3)))
    return image, label, masks

def parserMixed(record):
    keys_to_features = {"image": tf.io.FixedLenFeature([], tf.string),
                        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'label_beta': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        'label_gamma': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        'label_rates': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        'label_ratesEarly': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        # 'label_betaBoot': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        # 'label_gammaBoot': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                        'im_ind': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    height = tf.cast(parsed["height"], tf.int64)
    width = tf.cast(parsed["width"], tf.int64)
    im_ind = tf.cast(parsed["im_ind"], tf.int64)
    # label = tf.cast(parsed["label_beta"], tf.float32)
    label = tf.cast(parsed["label_gamma"], tf.float32)
    # label1 = tf.cast(parsed["label_rates"], tf.float32) + 1
    # label = tf.cast(parsed["label_ratesEarly"], tf.float32) + 1
    # label = tf.stack((label, label_gamma), axis=0)
    # label = label2 - label1
    image = tf.reshape(image, shape=[height, width, 3])
    masks = K.constant(np.ones((84, 84, 3)))
    return image, label, masks

def input_fn(filenames, batch_size=32, n_epoch=1, n_threads=4, f_augment=False, f_shuffle=True):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=n_threads)
    dataset = dataset.repeat()
    if f_shuffle:
        dataset = dataset.shuffle(4096)
    dataset = dataset.map(parserMixed, n_threads)
    # dataset = dataset.map(parser, n_threads)
    dataset = dataset.map(_resize, n_threads)
    dataset = dataset.map(_vgg_preproc, n_threads)
    # dataset = dataset.map(_crop, n_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=3)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    initializer = iterator.make_initializer(dataset)
    return iterator, initializer


def train_inputs_fn(b_size, n_epoch):
    return input_fn(filenames=["../data/trainPredSel224.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch), 28889

def test_inputs_fn(b_size, n_epoch):
    return input_fn(filenames=["../data/testPredSel224.tfrecords"],
                    batch_size=b_size, n_epoch=n_epoch, f_shuffle=False), 3099
