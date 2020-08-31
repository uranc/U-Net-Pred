from tensorflow.keras import applications
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Concatenate, LeakyReLU, ReLU, PReLU
from tensorflow.keras.initializers import Constant, Zeros
from model.pconv_layer import PConv2D, Conv2D
from model.losses import loss_fn, loss_fn_pred
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
import pdb
from model.losses import ssim_loss
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf
from keras import losses
from tensorflow.keras.layers import Dropout


def get_vgg(nrows, ncols, out_layer=None, is_trainable=False, inc_top=False):
    # make vgg percept up to pool 3
    vgg_model = applications.VGG16(
        weights="imagenet",
        include_top=inc_top,
        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    for out in layer_dict.keys():
        # print('hello')
        layer_dict[out].trainable = is_trainable

    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]

    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = is_trainable
    model.compile(loss='mse', optimizer='adam')
    return model

def encoder_partial_vgg(nets, masks, 
                        nfilts, nconv, bname, 
                        kernel_size=3, weights=None, use_inst=False):
    for i_conv in range(nconv):
        if weights:
            nets, masks = PConv2D(nfilts, kernel_size,
                                  activation=None,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=Constant(
                                      weights[2*i_conv+0]),
                                  bias_initializer=Constant(
                                      weights[2*i_conv+1]),
                                  name=bname+'_conv_'+str(i_conv+1),
                                  )([nets, masks])
            nets = ReLU()(nets)
        else:
            nets, masks = PConv2D(nfilts, kernel_size,
                                  activation=None,
                                  padding='same',
                                  use_bias=True,
                                  bias_initializer='zeros',
                                  kernel_initializer='he_normal',
                                  name=bname+'_conv_'+str(i_conv+1)
                                  )([nets, masks])
            nets = ReLU()(nets)
    cnets = nets
    cmasks = masks

    if not bname == 'vblock5':
        masks = MaxPooling2D((2, 2), strides=(
            2, 2), name='poolm_'+bname)(masks)
        nets = MaxPooling2D((2, 2), strides=(2, 2),
                            name='pooln_'+bname)(nets)
    return nets, masks, cnets, cmasks


def decoder_partial_vgg(nets, masks,
                        cnets, cmasks,
                        nfilts, nconv, bname):

    # upsample
    nets = UpSampling2D(size=(2, 2))(nets)
    masks = UpSampling2D(size=(2, 2))(masks)

    # concat
    nets = Concatenate(axis=-1)([nets, cnets])
    masks = Concatenate(axis=-1)([masks, cmasks])

    # # 2nd conv
    for i_conv in range(nconv):
        nets, masks = PConv2D(nfilts, 3,
                              activation=None,
                              padding='same',
                              use_bias=True,
                              bias_initializer='zeros',
                              kernel_initializer='he_normal',
                              name=bname+'_conv_'+str(i_conv+1)
                              )([nets, masks])
        nets = ReLU()(nets)
    return nets, masks

def make_pconv_uvgg(nets_in,
                    masks_in,
                    vgg_model=None):
    # encoder
    nets, masks, cnets_1, cmasks_1 = encoder_partial_vgg(
        nets_in, masks_in, 64, 2, 'vblock1', kernel_size=3)
    nets, masks, cnets_2, cmasks_2 = encoder_partial_vgg(
        nets, masks, 128, 2, 'vblock2', kernel_size=3)
    nets, masks, cnets_3, cmasks_3 = encoder_partial_vgg(
        nets, masks, 256, 3, 'vblock3', kernel_size=3)
    nets, masks, cnets_4, cmasks_4 = encoder_partial_vgg(
        nets, masks, 512, 3, 'vblock4', kernel_size=3)
    nets, masks, cnets_5, cmasks_5 = encoder_partial_vgg(
        nets, masks, 512, 3, 'vblock5', kernel_size=3)

    # decoder
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_4, cmasks_4, 512, 2, 'vblock7')
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_3, cmasks_3, 256, 2, 'vblock8')
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_2, cmasks_2, 128, 2, 'vblock9')
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_1, cmasks_1, 64, 2, 'vblock10')

    # output
    nets = Concatenate(axis=-1)([nets, nets_in])
    masks = Concatenate(axis=-1)([masks, masks_in])
    nets, masks = PConv2D(3, (3, 3),
                          activation=None,
                          padding='same',
                          use_bias=True,
                          bias_initializer='zeros',
                          kernel_initializer='he_normal',
                          name='block10')([nets, masks])
    return nets, masks

def build_compile_model(params):

    # load labels
    nr = params['input_size']
    nets_in = Input(shape=(nr, nr, 3))
    masks_in = Input(shape=(nr, nr, 3))

    # Settings
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # get vgg16 layers pool1, pool2, pool3
    vgg_model = get_vgg(nrows, ncols, out_layer=['block1_conv2',
                                                 'block2_conv2',
                                                 'block3_conv3',
                                                 'block4_conv3',
                                                 ])

    # Create UNet-like model
    outputs, masks = make_pconv_uvgg(nets_in,
                                     masks_in,
                                     vgg_model=vgg_model)

    # Setup the model inputs / outputs
    model = Model(inputs=[nets_in]+[masks_in], outputs=[outputs])
    total_loss = loss_fn(params, masks_in, vgg_model=vgg_model)
    ssim = ssim_loss(masks_in)

    # load weights i
    if params['weight_dir']:
        print('load')
        model.load_weights(params['weight_dir'])

    # compile
    model.compile(optimizer=optimizers.Adam(lr=params['lr']),
                  loss=total_loss, metrics=[ssim])
    return model
