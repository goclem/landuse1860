#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Optimises models for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import gc
import numpy as np
import pandas as pd
import tensorflow

from matplotlib import colors, pyplot
from numpy import random
from landuse1860_model import unet_model
from landuse1860_utilities import *
from tensorflow.keras import backend, callbacks, layers, losses, metrics, models, optimizers, utils

# Utilities
classes = dict(zip(['undefined', 'buildings', 'transports', 'crops', 'meadows', 'pastures', 'specialised', 'forests', 'water'], np.arange(9)))
print('GPU Available:', bool(len(tensorflow.config.experimental.list_physical_devices('GPU'))))

#%% INITIALISES GENERATOR

class data_generator(utils.Sequence):
    
    def __init__(self, image_files:list, label_files:list, n_files:str, batch_size:int, target_class:int=None) -> tuple:
        self.image_files  = image_files
        self.label_files  = label_files
        self.n_files      = n_files
        self.batch_size   = batch_size
        self.n_outputs    = 9
        self.block_size   = (256, 256)
        self.target_class = target_class
        self.indices      = random.permutation(np.arange(len(self.label_files)))
    
    def __len__(self):
        return len(self.label_files) // self.n_files

    def __getitem__(self, file_index):
        # Samples sequences
        file_indices = self.indices[file_index * self.n_files:(file_index + 1) * self.n_files]
        # print(f"Loading: {', '.join(mapids(self.label_files[file_indices]))}")
        # Loads images and labels
        batch_labels = np.array([read_raster(file, dtype=int) for file in self.label_files[file_indices]])
        batch_labels = images_to_blocks(batch_labels, block_size=self.block_size, mode='constant', constant_values=0)
        batch_images = np.array([read_raster(file, dtype=int) for file in self.image_files[file_indices]])
        batch_images = images_to_blocks(batch_images, block_size=self.block_size, mode='constant', constant_values=255)
        # Keeps blocks >50% filled
        keep_indices = np.sum(batch_labels!=0, axis=(1,2,3)) / np.prod(self.block_size) >= 0.5
        batch_labels = batch_labels[keep_indices]
        batch_images = batch_images[keep_indices]
        # Formats images
        batch_images = np.where(batch_labels==0, 255, batch_images)
        batch_images = batch_images / 255
        # Formats labels
        if self.target_class is None:
            batch_labels = utils.to_categorical(batch_labels, num_classes=self.n_outputs)
        else:
            batch_labels = np.where(batch_labels==self.target_class, 1, 0).astype(float)
            keep_indices = np.sum(batch_labels, axis=(1,2,3)) >= 0.01
            batch_labels = batch_labels[keep_indices]
            batch_images = batch_images[keep_indices]
        # Shuffles samples
        rand_indices = random.choice(np.arange(len(batch_labels)), self.batch_size, replace=True)
        batch_images = batch_images[rand_indices]
        batch_labels = batch_labels[rand_indices]
        return batch_images, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Training tiles
training = [filename(file).replace('label_', '') for file in search_data(paths['labels'], 'tif$')]
training = f"({'|'.join(training)})\.tif$"

# Loads images and labels
image_files = search_data(paths['images'], pattern=training)
label_files = search_data(paths['labels'], pattern=training)
assert np.array_equal(mapids(image_files), mapids(label_files))

# Splits samples
samples = dict(train=0.70, valid=0.15, test=0.15)
images_train, images_valid, images_test = sample_split(image_files, sizes=samples, seed=0)
labels_train, labels_valid, labels_test = sample_split(label_files, sizes=samples, seed=0)
del image_files, label_files

#%% ESTIMATES BASE MODELS

# Initialises generator
train_generator = data_generator(images_train, labels_train, n_files=2, batch_size=64,  target_class=None)
valid_generator = data_generator(images_valid, labels_valid, n_files=4, batch_size=256, target_class=None)
test_generator  = data_generator(images_test,  labels_test,  n_files=4, batch_size=256, target_class=None)

''' Checks generator
batch_images, batch_labels = train_generator.__getitem__(0)
for i in random.choice(range(len(batch_images)), 5, replace=False):
    compare(images=[batch_images[i], batch_labels[i,...,1]], titles=['Image', 'Label'])
del batch_images, batch_labels, i
'''

# Initialises model
model = unet_model(input_shape=(256, 256, 3), n_outputs=len(classes), filters=16, output_activation='softmax', dropout=0.1, montecarlo=True, label='base')
model.summary()

# Transfers latest base model parameters
params = search_data(paths['models'], pattern='base\.h5$')
params = params[np.argmax([os.stat(file).st_birthtime for file in params])]
params = models.load_model(params).get_weights()
model.set_weights(params)
del params

# Compiles models
model.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), 
              loss=losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False), 
              metrics=[metrics.CategoricalAccuracy()])

# Optimises model
training = model.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch =len(train_generator.label_files) // train_generator.n_files,
    validation_steps=len(valid_generator.label_files) // valid_generator.n_files,
    epochs=100,
    callbacks=callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    verbose=1
)

# Saves model and training history
models.save_model(model, f"{paths['models']}/{model.name}_base.h5")

#%% ESTIMATES SPECIFIC MODELS

# Defines target class
for target in ['buildings', 'transports', 'crops', 'meadows', 'pastures', 'forests', 'water', 'specialised']:
    
    print(f'Processing: {target}')
    
    # Initialises generator
    train_generator = data_generator(images_train, labels_train, n_files=4, batch_size=64,  target_class=classes[target])
    valid_generator = data_generator(images_valid, labels_valid, n_files=4, batch_size=256, target_class=classes[target])
    test_generator  = data_generator(images_test,  labels_test,  n_files=4, batch_size=256, target_class=classes[target])
    
    # Initialises model
    model = unet_model(input_shape=(256, 256, 3), n_outputs=1, filters=16, output_activation='sigmoid', dropout=0.1, montecarlo=True, label=target)
    # model.summary()

    # Transfers model parameters...
    params = search_data(paths['models'], pattern=f'(base|{target})\.h5$')
    index  = np.char.find(params, target) != -1
    if np.any(index): params = params[index]
    params = params[np.argmax([os.stat(file).st_birthtime for file in params])]
    params = models.load_model(params)

    if np.any(index): # ... from the target model
        model.set_weights(params.get_weights())
    else: # ... from the base model
        for i in range(len(model.layers)-1):
            model.layers[i].set_weights(params.layers[i].get_weights())
        model.layers[-1].set_weights([np.expand_dims(weights[..., classes[target]], -1) for weights in params.layers[-1].get_weights()])
    del params, index

    # Compiles models
    model.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), 
                loss=losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False), 
                metrics=[metrics.Precision(), metrics.Recall()])

    # Optimises model
    training = model.fit(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch =len(train_generator.label_files) // train_generator.n_files,
        validation_steps=len(valid_generator.label_files) // valid_generator.n_files,
        epochs=100,
        callbacks=callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        verbose=1
    )

    # Saves model
    models.save_model(model, f"{paths['models']}/{model.name}_{target}.h5")
    
    # Clears memory
    del target, model, training
    backend.clear_session()
    gc.collect()

#%% FINAL MODEL

def final_model(input_shape:dict, n_outputs:int):
    # Loads models
    target_models = list(classes.keys())[1:]
    target_models = [models.load_model(search_data(paths['models'], pattern=f'{target}.h5$')[0]) for target in target_models]
    for model in target_models:
        model.trainable = False
    # Combines models
    inputs  = layers.Input(input_shape, name='input')
    probas  = [model(inputs) for model in target_models]
    probas  = layers.concatenate([inputs] + probas, axis=3, name='concatenate')
    probas  = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='transform')(probas)
    outputs = layers.Conv2D(filters=n_outputs, kernel_size=(1, 1), padding='same', activation='softmax', name='output')(probas)
    model   = models.Model(inputs=inputs, outputs=outputs, name='model_final')
    return model

# Initialises generator
train_generator = data_generator(images_train, labels_train, n_files=2, batch_size=64,  target_class=None)
valid_generator = data_generator(images_valid, labels_valid, n_files=4, batch_size=256, target_class=None)
test_generator  = data_generator(images_test,  labels_test,  n_files=4, batch_size=256, target_class=None)

# Initialises model
model = final_model(input_shape=(256, 256, 3), n_outputs=len(classes))
model.summary()

# Transfers latest final model parameters
params = search_data(paths['models'], pattern='final\.h5$')
params = params[np.argmax([os.stat(file).st_birthtime for file in params])]
params = models.load_model(params).get_weights()
model.set_weights(params)
del params

# Computes class weights
alphas = np.zeros((len(classes),), dtype=int)
for label in search_data(paths['labels'], pattern=training):
    freqs = read_raster(label, dtype=int)
    freqs = np.bincount(freqs.flatten(), minlength=len(classes))
    alphas += freqs
alphas = 1 / np.sqrt(alphas)
alphas = alphas / np.sum(alphas)
del freqs, label

# Compiles models
model.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), 
              loss=losses.CategoricalFocalCrossentropy(alpha=alphas, gamma=2.0, from_logits=False), 
              metrics=[metrics.CategoricalAccuracy()])

# Optimises model
training = model.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch =len(train_generator.label_files) // train_generator.n_files,
    validation_steps=len(valid_generator.label_files) // valid_generator.n_files,
    epochs=100,
    callbacks=callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    verbose=1
)

# Saves model and training history
models.save_model(model, f"{paths['models']}/{model.name}.h5")

#%% EVALUATES MODELS

# Loads model
model = 'base'
model = search_data(paths['models'], pattern=f'{model}\.h5$')
model = model[np.argmax([os.stat(file).st_birthtime for file in model])]
model = models.load_model(model)

''' Computes test error
performance = model.evaluate(test_generator, steps=len(test_generator.label_files) // test_generator.n_files)
print('Test loss: {:.4f}\nTest accuracy: {:.4f}'.format(*performance))
del performance
'''

# Checks predictions
batch  = random.choice(len(test_generator.label_files) // test_generator.n_files)
images, labels = test_generator.__getitem__(batch)
probas = model.predict(images)

# Formats labels and predictions
if probas.shape[-1] > 1:
    predicts = np.argmax(probas, axis=3, keepdims=True)
    labels   = np.argmax(labels, axis=3, keepdims=True)
else:
    predicts = np.where(probas > 0.5, classes[target], 0)
    labels   = np.where(labels == 1,  classes[target], 0)

# Figure
cmap  = colors.ListedColormap(['#ffffff', '#e31a1c', '#000000', '#fdd8ac', '#a6cee3', '#b2df8a', '#b0b0b0', '#33a02c', '#1f78b4'], name='landuse')
norm  = colors.BoundaryNorm(np.arange(10), cmap.N)
index = random.choice(np.arange(test_generator.batch_size), 5, replace=False)
for image, label, proba, predict in zip(images[index], labels[index], probas[index], predicts[index]):
    fig, axs = pyplot.subplots(nrows=1, ncols=4, figsize=(3*10, 10))
    axs[0].imshow(image)
    axs[1].imshow(label,   cmap=cmap, norm=norm)
    axs[2].imshow(predict, cmap=cmap, norm=norm)
    # axs[2].imshow(proba,   cmap='viridis')
    for ax in axs.ravel(): ax.set_axis_off()
    pyplot.tight_layout()
    pyplot.show()

del batch, images, labels, predicts, index, cmap, norm, image, label, predict

# %%
