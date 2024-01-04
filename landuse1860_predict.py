#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Predicts maps for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import concurrent.futures as future
import geopandas as gpd
import itertools
import numpy as np
import pandas as pd
import tensorflow

from landuse1860_utilities import *
from landuse1860_models import final_model
from tensorflow.keras import layers, models, preprocessing

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', bool(len(tensorflow.config.experimental.list_physical_devices('GPU'))))

# Utilities
classes = dict(zip(['undefined', 'buildings', 'transports', 'crops', 'meadows', 'pastures', 'specialised', 'forests', 'water', 'border'], np.arange(10)))

#%% COMPUTES PREDICTIONS

def predict_probas(srcfile:str, n_outputs:int=10) -> np.ndarray:
    '''Computes moving window predictions for a map'''
    original = read_raster(srcfile) / 255
    shifted  = np.pad(original, ((128, 128), (128, 128), (0, 0)), mode='constant', constant_values=1)
    probas   = list()
    for i, image in enumerate([original, shifted]):
        blocks = images_to_blocks(np.expand_dims(image, axis=0), block_size=(256, 256), mode='constant', constant_values=1)
        subset = np.array(list(map(lambda x: not_empty(x, value=1), blocks)))
        proba  = np.zeros(blocks.shape[:3] + (n_outputs,))
        proba[subset] = model.predict(blocks[subset], verbose=0)
        proba  = blocks_to_images(proba, (1,) + image.shape[:2] + (n_outputs,))
        if i == 1: proba = proba[:,128:-128, 128:-128,:]
        probas.append(proba)
    probas = np.mean(probas, axis=0)
    return probas

# Loads final model
model = final_model(input_shape=(256, 256, 3), n_outputs=len(classes))
model.load_weights(f"{paths['models']}/model32_final.ckpt")

# Paths
srcfiles = search_data(paths['images'], pattern='tif$')
dstfiles = np.array([f"{paths['predictions']}/predict_{mapid(srcfile)}.tif" for srcfile in srcfiles])
subset   = ~np.vectorize(path.exists)(dstfiles)

# Computes probabilities
for i, (srcfile, dstfile) in enumerate(zip(srcfiles[subset], dstfiles[subset])):
    print(f'Processing ({i+1:03d}/{np.sum(subset):03d}): {mapid(srcfile)}')
    predict = predict_probas(srcfile)
    predict = np.argmax(predict, axis=-1, keepdims=True)
    write_raster(np.squeeze(predict), srcfile, dstfile, nodata=None, dtype='uint8')
del predict_probas, srcfiles, dstfiles, subset, i, srcfile, dstfile, predict

#%% POSTPROCESSES PREDICTIONS

def postprocess(srcfile:str, dstfile:str) -> None:
    '''Postprocesses predictions'''
    print(f'Processing: {mapid(srcfile)}')
    predict = read_raster(srcfile)
    predict = np.where(rasterise(france, profile=srcfile), predict, 0)
    predict = np.where(rasterise(water, profile=srcfile), classes['water'], predict)
    predict = np.where(rasterise(transports, profile=srcfile), classes['transports'], predict)
    predict = np.where(rasterise(buildings, profile=srcfile), classes['buildings'], predict)
    write_raster(predict, srcfile, dstfile, dtype='uint8')

# Loads country mask
france = gpd.read_file(f"{paths['vectors']}/france.gpkg")
france = france.to_crs(2154)

# Loads building vectors
buildings  = pd.concat((
    gpd.read_file(f"{paths['vectors']}/fixes/buildings1860_paris.gpkg")[['geometry']], 
    gpd.read_file(f"{paths['vectors']}/fixes/buildings1860_fixes.gpkg")
))

# Loads transports vectors
transports = pd.concat((
    gpd.read_file(f"{paths['vectors']}/fixes/roads1860.gpkg"),
    gpd.read_file(f"{paths['vectors']}/fixes/rails1860.gpkg")
))
transports.geometry = transports.geometry.buffer(4)

# Loads water vectors
rivers = gpd.read_file(f"{paths['vectors']}/fixes/rivers1860.gpkg")[['geometry']]
canals = gpd.read_file(f"{paths['vectors']}/fixes/canals1860.gpkg")
canals.geometry = canals.geometry.buffer(4)
water = pd.concat((rivers, canals))
del rivers, canals

# Paths
srcfiles = search_data(paths['predictions'], pattern='tif$')
dstfiles = np.array([f"{paths['desktop']}/postprocessed/predict_{mapid(srcfile)}.tif" for srcfile in srcfiles])
subset   = ~np.vectorize(path.exists)(dstfiles)

# Postprocesses predictions
with future.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(postprocess, srcfiles[subset], dstfiles[subset])
del postprocess, france, buildings, transports, water, srcfiles, dstfiles, subset

#%% AGGREGATES PREDICTIONS
