"""Tensorflow utility functions for training"""
import logging
import os
from tqdm import trange
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from model.utils import save_dict_to_json
from model.callback_fn import TensorBoardCustom
from tqdm import trange
from model.utils import save_dict_to_json


def evaluate_sess(model, num_steps, current_epoch, save_dir):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    callbacks.append(TensorBoardCustom(log_dir=save_dir,
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=True,
                                       update_freq='epoch'))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.hdf5',
                                        monitor='loss',
                                        verbose=1,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        mode='auto',
                                        period=1))
    # Get relevant graph operations or nodes needed for training
    return model.fit(steps_per_epoch=num_steps,
                     initial_epoch=current_epoch,
                     epochs=current_epoch+2,
                     verbose=1,
                     callbacks=callbacks)
