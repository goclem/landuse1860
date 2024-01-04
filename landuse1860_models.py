#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Models for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import numpy as np
from tensorflow.keras import callbacks, layers, initializers, models, utils
from landuse1860_utilities import classes

#%% UNET MODEL

# Convolution block
def convolution_block(input, filters:int, dropout:float, training:bool, name:str):
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution1')(input)
    x = layers.Activation(activation='relu', name=f'{name}_activation1')(x)
    x = layers.BatchNormalization(name=f'{name}_normalisation1')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution2')(x)
    x = layers.Activation(activation='relu', name=f'{name}_activation2')(x)
    x = layers.BatchNormalization(name=f'{name}_normalisation2')(x)
    x = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(x, training=training)
    return x

# Encoder block
def encoder_block(input, filters:int, dropout:float, training:bool, name:str):
    x = convolution_block(input=input, filters=filters, dropout=dropout, training=training, name=name)
    p = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(x)
    return x, p

# Decoder block
def decoder_block(input, skip, filters:int, dropout:float, training:bool, name:str):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', name=f'{name}_transpose')(input)
    x = layers.Concatenate(name=f'{name}_concatenate')([x, skip])
    x = convolution_block(input=x, filters=filters, dropout=dropout, training=training, name=name)
    return x

# Decoder block
def unet_model(input_shape:dict, n_outputs:int, filters:int, output_activation:str, dropout:float, montecarlo:bool, label:str, seed:int=0):
    # Input
    inputs = layers.Input(input_shape, name='input')
    # Encoder path
    s1, p1 = encoder_block(input=inputs, filters=1*filters, dropout=dropout, training=montecarlo, name='encoder1')
    s2, p2 = encoder_block(input=p1,     filters=2*filters, dropout=dropout, training=montecarlo, name='encoder2')
    s3, p3 = encoder_block(input=p2,     filters=4*filters, dropout=dropout, training=montecarlo, name='encoder3')
    s4, p4 = encoder_block(input=p3,     filters=8*filters, dropout=dropout, training=montecarlo, name='encoder4')
    # Bottleneck
    b1 = convolution_block(input=p4, filters=16*filters, dropout=dropout, training=montecarlo, name='bottleneck')
    # Decoder path
    d1 = decoder_block(input=b1, skip=s4, filters=8*filters, dropout=dropout, training=montecarlo, name='decoder1')
    d2 = decoder_block(input=d1, skip=s3, filters=4*filters, dropout=dropout, training=montecarlo, name='decoder2')
    d3 = decoder_block(input=d2, skip=s2, filters=2*filters, dropout=dropout, training=montecarlo, name='decoder3')
    d4 = decoder_block(input=d3, skip=s1, filters=1*filters, dropout=dropout, training=montecarlo, name='decoder4')
    # Output
    outputs = layers.Conv2D(filters=n_outputs, kernel_size=(1, 1), padding='same', activation=output_activation, name='output')(d4)
    # Model
    model = models.Model(inputs=inputs, outputs=outputs, name=f"unet{filters}_{label}")
    return model

#%% FINAL MODEL

def init_probas(input_shape:dict):
    target_models = [models.load_model(f'../data_scem/models/unet32_{target}.h5') for target in list(classes.keys())[1:]]
    inputs  = layers.Input(input_shape, name='input')
    probas  = [model(inputs) for model in target_models]
    outputs = layers.concatenate(probas, axis=3, name='concatenate')
    model   = models.Model(inputs=inputs, outputs=outputs, name='model32_probas')
    return model

def final_model(input_shape:dict, n_outputs:int):
    model_probas = init_probas(input_shape)
    model_probas.trainable = False
    inputs  = layers.Input(input_shape, name='input')
    probas  = model_probas(inputs)
    probas  = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='conv1')(probas)
    probas  = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='conv2')(probas)
    probas  = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='conv3')(probas)
    outputs = layers.Conv2D(filters=n_outputs, kernel_size=(1, 1), padding='same', activation='softmax', name='output')(probas)
    model   = models.Model(inputs=inputs, outputs=outputs, name='model32_final')
    return model