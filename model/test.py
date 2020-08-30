from model.model_fn import build_compile_model
from model.training import train_sess
import os
import pdb
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks as cb
from shutil import copyfile
from distutils.file_util import copy_file
from distutils.dir_util import copy_tree
from model.input_fn import train_inputs_fn, valid_inputs_fn, test_inputs_fn
from model.search_hyperparameters import farm_hyperparameters

# params = {'weight_dir': 'experiments/l1/weights.last.h5', 'lr': 5e-2,
#           'p_content': 1e-3, 'p_style': 1e-3, 'p_hole': 5/12.25, 'p_tv': 0,
#           'b_size': 32, 'current_epoch': 9,
#           'n_epoch': 150, 'exp_dir': 'experiments/l1',
# a          'inst_norm': True}
# params = {'weight_dir': 'experiments/l2/weights.last.h5', 'lr': 1e-3,
#           'p_content': 0.0345, 'p_style': 100, 'p_hole': 5/12.25, 'p_tv': 0.0345,
#           'b_size': 32, 'current_epoch': 73,
#           'n_epoch': 150, 'exp_dir': 'experiments/l2',
#           'inst_norm': True}
# 'p_content': 0.0345, 'p_style': 100, 'p_hole': 5/12.25, 'p_tv': 0.0345,
params = {'weight_dir': '', 'lr': 6.89*1e-4,
          'p_content': 0, 'p_style': 0, 'p_hole': 0, 'p_tv': 0,
          'p_fft_s': 0.006, 'p_fft_c': 130.24,  # 'p_fft_s': 0.330, 'p_fft_c': 505.47, 
          'p_fft_abs': 1, 'p_fft_log': 0.022, 'p_fft_phase': 6.61e-05, # .0556
          'b_size': 16, 'current_epoch': 0,
          'n_epoch': 150000,
          # 'exp_dir': 'experiments/l1',
          # 'exp_dir': 'experiments/l2',
          # 'exp_dir': 'experiments/l5',
          # 'exp_dir': 'experiments/l6',
          # 'exp_dir': 'experiments/l7',
          # 'exp_dir': 'experiments/l8',
          'exp_dir': 'experiments/l10',
          'inst_norm': False,
          'worker' : args.worker,
          # 'n_workers': args.n_workers
          # 'n_workers': 4
          }
print(params['b_size'])
# assert (params['weight_dir'] is not None) \
#     and (params['current_epoch'] == 0), 'forgetting the weights?'

# training
mode = 'train'

# save scripts
copy_tree('./model', params['exp_dir']+'/model')
copy_file('./test.py', params['exp_dir']+'/model/test.py')

# inputs
train_inputs, train_size = \
    train_inputs_fn(params['b_size'], params['n_epoch'])
valid_inputs, valid_size = \
    valid_inputs_fn(params['b_size'], params['n_epoch'])
train_steps = int(np.floor(train_size/params['b_size']))
valid_steps = int(np.floor(4096*4/params['b_size']))

# # model
model = build_compile_model(mode, params)
model.summary()

# # train
hist = train_sess(model,
                  train_inputs,
                  train_steps,
                  valid_inputs,
                  valid_steps,
                  params['current_epoch'],
                  params['n_epoch'],
                  params['exp_dir'])

# sess = tf.Session()
# a,b,c = sess.run(train_inputs.get_next())
# hist = farm_hyperparameters(mode, params)


# min_loss = 1000000000
# for i_ep in range(params['n_epoch']):

#     # new params
# params['current_epoch'] += 1
#     params['p_content'] = new_params['p_content']
#     params['p_style'] = new_params['p_style']
# if ((i_ep+1) % 3000) == 0:
#     K.set_value(model.optimizer.lr, params['lr']/10)
# training
# model, input_size, init = build_compile_model(mode, params)
# print(model.summary())
# train_steps = int(np.floor(input_size/params['b_size']))
# train_steps = 1

# sess = K.get_session()
# sess.run(initop)
# tf.contrib.saved_model.save_keras_model
# tf.contrib.saved_model.load_keras_model
# # break
# K.clear_session()

# # evaluate
# mode = 'hyper'
# hist = evaluate_epoch(mode, params)
# eval_loss = hist.history['loss'][0]
# if eval_loss < min_loss:
#     min_loss = eval_loss
#     copyfile(params['exp_dir'] + '/weights.last.h5',
#              params['exp_dir'] + '/weights.ep'+str(params['current_epoch'])+'_best.h5')

# # clean
# mode = 'hyper_train'
# K.clear_session()

# # evaluation
# if (i_ep+1) % 5 == 0:
#     mode = 'hyper'
#     model, input_size = build_compile_model(mode, params)
#     new_params, eval_loss = farm_hyperparameters(mode, params)

#     # save best eval model
#     if eval_loss < min_loss:
#         min_loss = eval_loss
#         copyfile(params['exp_dir'] + '/weights.last.h5',
#                  params['exp_dir'] + '/weights.ep'+str(params['current_epoch'])+'_best.h5')

#     # back to training
# mode = 'hyper_train'
# K.clear_session()


# test_filenames = os.path.join(data_dir, "test50.tfrecords")
# test_inputs, test_size = test_inputs_fn(params['b_size'], params['n_epoch'])
