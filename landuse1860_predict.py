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
from tensorflow.keras import layers, models, preprocessing

# TensorFlow
print('TensorFlow version:', tensorflow.__version__)
print('GPU Available:', bool(len(tensorflow.config.experimental.list_physical_devices('GPU'))))

#%% PREDICTS PROBABILITIES

def predict_probas(srcfile:str, n_outputs:int=9) -> np.ndarray:
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

# Loads model
model = search_data(paths['models'], pattern='final\.h5$')
model = model[np.argmax([os.stat(file).st_birthtime for file in model])]
model = models.load_model(model)

# Paths
srcfiles = search_data(f"{paths['images']}", pattern='tif$')
dstfiles = np.array([f"{paths['predictions']}/predict_{mapid(srcfile)}.tif" for srcfile in srcfiles])
subset   = ~np.vectorize(path.exists)(dstfiles)
subset   = np.char.find(srcfiles, '0800_6500') >= 0 # Testing

# Computes probabilities
for i, (srcfile, dstfile) in enumerate(zip(srcfiles[subset], dstfiles[subset])):
    print(f'Processing ({i+1:03d}/{np.sum(subset):03d}): {mapid(srcfile)}')
    predict = predict_probas(srcfile)
    predict = np.argmax(predict, axis=-1, keepdims=True)
    write_raster(np.squeeze(predict), srcfile, dstfile, nodata=None, dtype='uint8')
del predict_probas, srcfiles, dstfiles, subset, i, srcfile, dstfile, predict

#%% COMPUTES LABELS

# Sets thresholds
thresholds = dict(south=0.43, north=0.47, london=0.48)

#! Set estimated threshold
def compute_labels(srcfile:str, dstfile:str, threshold:float):
    '''Computes labels from probabilities'''
    print(f'Processing: {mapid(srcfile)}')
    os.system('gdal_calc.py --overwrite -A {srcfile} --outfile={dstfile} --calc="A>={threshold}" --type=Byte --quiet'.format(srcfile=srcfile, dstfile=dstfile + '.tif', threshold=threshold))
    polygonise(dstfile + '.tif', dstfile + '.gpkg', cellvalue=1)
    os.remove(dstfile + '.tif')
    # Adds map identifier
    predict = gpd.read_file(dstfile + '.gpkg').drop(columns=['value'])
    predict['mapid'] = mapid(srcfile)
    predict.to_file(dstfile + '.gpkg', driver='GPKG')
    
# Paths
srcfiles = search_data(f"{paths['temporary']}/data_{region}", pattern='proba\.tif$')
dstfiles = np.array([f"{paths['temporary']}/data_{region}/{mapid(srcfile)}_predict" for srcfile in srcfiles])
subset   = ~np.vectorize(path.exists)(np.char.add(dstfiles, '.gpkg'))

# Computes labels
with future.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(compute_labels, srcfiles[subset], dstfiles[subset], itertools.repeat(thresholds[region]))
del compute_labels, srcfiles, subset, executor

#%% AGGREGATES LABELS

# Works despite error message
os.system('ogrmerge.py -single -overwrite_ds -f GPKG -nln {outlayer} -o {outfile} {pattern}'.format(outlayer=f'predictions_{region}', outfile=f"{paths[region]}/predictions_{region}.gpkg", pattern=f"{paths['temporary']}/data_{region}/*.gpkg"))
# [os.remove(dstfile + '.gpkg') for dstfile in dstfiles]
del dstfiles

#%% FIXES PREDICTIONS #! Run carefully

# Removes buildings from areas not properly exluded by the RAs
predict = gpd.read_file(f"{paths['north']}/predictions_north.gpkg")
masks   = gpd.read_file(f"{paths['vectors']}/extents_north.gpkg")
masks   = masks.dissolve('mapid').reset_index()

fixed = list()
for mapid in predict.mapid.unique():
    print(f'Processing: {mapid}')
    mask   = masks[masks.mapid==mapid]
    subset = predict[predict.mapid==mapid]
    index  = np.array([mask.intersects(point) for point in subset.centroid]).flatten()
    fixed.append(subset[~index])

pd.concat(fixed).to_file(f"{paths['north']}/predictions_north.gpkg", driver='GPKG')
    
# Manages the overlap between the map collections
predict = gpd.read_file(f"{paths['south']}/predictions_south.gpkg")
pattern = '|'.join(mapids(search_data(paths['north'], pattern='image.tif'))) + '|london'
masks = gpd.read_file(f"{paths['vectors']}/intersections.gpkg")
masks = masks[np.array([len(re.findall(pattern, mapids)) == 1 for mapids in masks.mapids])]
masks['mapid'] = masks.mapids.str.replace('_.*$', '')
del pattern

fixed = list()
for mapid in predict.mapid.unique():
    print(f'Processing: {mapid}')
    subset = predict[predict.mapid==mapid]
    index  = np.zeros(len(subset), dtype=bool)
    mask   = masks[masks.mapid==mapid].dissolve(by='mapid')
    if len(mask) > 0: 
        index = np.array([mask.intersects(point) for point in subset.centroid]).flatten()
    fixed.append(subset[~index])

pd.concat(fixed).to_file(f"{paths['south']}/predictions_south.gpkg", driver='GPKG')