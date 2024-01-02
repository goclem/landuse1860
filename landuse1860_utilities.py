#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Modules
import geopandas as gpd
import math
import numpy as np
import pandas as pd
import os
import rasterio
import re
import shutil

from rasterio import features, profiles, transform, windows
from numpy import random
from matplotlib import pyplot
from os import path
from tensorflow.keras import utils

#%% PATHS UTILITIES

home  = path.expanduser('~')
paths = dict(
    data='../data_scem',
    images='../data_scem/images',
    labels='../data_scem/labels',
    masks='../data_scem/masks',
    segments='../data_scem/segments',
    predictions='../data_scem/predictions',
    models='../data_scem/models',
    vectors='../data_scem/vectors',
    utilities='../data_scem/utilities',
    desktop=f'{home}/Desktop',
    temporary=f'{home}/Desktop/temporary'
)
del home

#%% FILE UTILITIES

def search_data(directory:str='../data', pattern:str='.*') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    files = np.array(files)
    return files

def reset_folder(path:str, remove:bool=False) -> None:
    '''Resets a folder'''
    if remove:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

def mapid(srcfile:str) -> str:
    '''Extracts the madpid from a file name'''
    return re.findall('[0-9]{4}_[0-9]{4}', filename(srcfile))[0]

mapids = np.vectorize(mapid)

def filename(filepath:str, extension=False) -> str:
    '''Extracts file name'''
    filename = os.path.basename(filepath)
    if extension is False:
        filename = os.path.splitext(filename)[0]
    return filename

filenames = np.vectorize(filename)

def dirname(filepath:str, extension=False) -> str:
    '''Extracts file name'''
    dirname = os.path.basename(os.path.dirname(filepath))
    return dirname

dirnames  = np.vectorize(dirname)

#%% ARRAY UTILITIES

def sample_split(images:np.ndarray, sizes:dict, seed:int=0) -> list:
    '''Splits the data multiple samples'''
    random.seed(seed)
    samples = list(sizes.keys())
    indexes = random.choice(samples, images.shape[0], p=list(sizes.values()))
    samples = [images[indexes == sample, ...] for sample in samples]
    return samples

def frequency(array):
    '''Computes the frequency of each value in an array'''
    values, counts = np.unique(array, return_counts=True)
    output = pd.Series(counts, name='counts', index=values)
    return output

def images_to_blocks(images:np.ndarray, block_size:tuple=(256, 256), shift:bool=False, mode:str='constant', constant_values:int=None) -> np.ndarray:
    '''Converts images to blocks of a given size'''
    # Initialises quantities
    nimages, imagewidth, imageheight, nbands = images.shape
    blockwidth, blockheight = block_size
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Reshape images into blocks
    images = np.pad(images, ((0, 0), (padwidth, padwidth), (padheight, padheight), (0, 0)), mode=mode, constant_values=constant_values)
    blocks = images.reshape(nimages, nblockswidth, blockwidth, nblocksheight, blockheight, nbands).swapaxes(2, 3)
    blocks = blocks.reshape(-1, blockwidth, blockheight, nbands)
    return blocks

def blocks_to_images(blocks:np.ndarray, image_size:tuple, shift:bool=False) ->  np.ndarray:
    '''Converts blocks to images of a given size'''
    # Initialises quantities
    nimages, imagewidth, imageheight, nbands = image_size
    blockwidth, blockheight = blocks.shape[1:3]
    nblockswidth  = (imagewidth  // blockwidth  + 1 + shift)
    nblocksheight = (imageheight // blockheight + 1 + shift)
    # Defines padding
    padwidth  = int(((nblockswidth)  * blockwidth  - imagewidth)  / 2)
    padheight = int(((nblocksheight) * blockheight - imageheight) / 2)
    # Converts blocks into images
    images = blocks.reshape(-1, nblockswidth, nblocksheight, blockwidth, blockheight, nbands).swapaxes(2, 3)
    images = images.reshape(-1, (imagewidth + (2 * padwidth)), (imageheight + (2 * padheight)), nbands)
    images = images[:, padwidth:imagewidth + padwidth, padheight:imageheight + padheight, :]
    return images

def not_empty(image, value:int=255, type:str='all'):
    '''Checks for empty images'''
    test = np.equal(image, np.full(image.shape, value))
    if type == 'all': test = np.all(test)
    if type == 'any': test = np.any(test)
    test = np.invert(test)
    return test

#%% RASTER UTILITIES

def read_raster(source:str, band:int=None, window=None, dtype:str='int') -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='uint8') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def rasterise(source, profile, layer:str=None, varname:str=None, all_touched:bool=False, update:dict=None) -> np.ndarray:
    '''Rasterises a vector file'''
    if isinstance(source, str):
        source = gpd.read_file(source, layer=layer)
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    if update is not None: 
        profile.update(count=1, **update)
    geometries = source['geometry']
    if varname is not None: 
        geometries = zip(geometries, source[varname])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'], dtype=profile['dtype'], all_touched=all_touched)
    image = np.expand_dims(image, 2)
    return image

def polygonise(srcfile:str, dstfile:str, band:int=1, varname:str='value', cellvalue:int=None) -> gpd.GeoDataFrame:
    '''Polygonises a raster band'''
    source   = rasterio.open(srcfile)
    polygons = source.read(band)
    if cellvalue is not None:
        polygons = list({'properties': {varname: value}, 'geometry': shape} for i, (shape, value) in enumerate(features.shapes(polygons, transform=source.transform)) if value == cellvalue)
    else:
        polygons = list({'properties': {varname: value}, 'geometry': shape} for i, (shape, value) in enumerate(features.shapes(polygons, transform=source.transform)))
    polygons = gpd.GeoDataFrame.from_features(polygons, crs=source.crs)
    if dstfile is not None:
        polygons.to_file(dstfile, driver='GPKG')
        return None
    else:
        return polygons
    
def profile_from_extent(source, resolution:tuple=None, dtype:str=None) -> None:
    '''Creates a raster from the extent of a geopandas object'''
    if isinstance(source, str): 
        source = gpd.read_file(source)
    xmin, ymin, xmax, ymax = source.geometry.total_bounds
    xres, yres = resolution
    xsize   = int((xmax - xmin) // xres) + 1
    ysize   = int((ymax - ymin) // yres) + 1
    profile = profiles.Profile(
        driver='GTiff',
        width=xsize,
        height=ysize,
        count=1,
        dtype=dtype,
        crs=source.crs,
        transform=transform.Affine(xres, 0, xmin, 0, -yres, ymax))
    return profile

#%% DISPLAY UTILITIES

def display(image:np.ndarray, title:str='', path:str=None, cmap:str='gray', figsize=(10, 10), fontsize:int=20, dpi:int=300) -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=fontsize)
    ax.set_axis_off()
    pyplot.tight_layout()
    if path is not None:
        pyplot.savefig(path, dpi=dpi)
    else:
        pyplot.show()

def compare(images:list, titles:list=[''], cmaps:list=['gray'], path:str=None) -> None:
    '''Displays multiple images'''
    nimage = len(images)
    if len(titles) == 1:
        titles = titles * nimage
    if len(cmaps) == 1:
        cmaps = cmaps * nimage
    fig, axs = pyplot.subplots(nrows=1, ncols=nimage, figsize=(10 * nimage, 10))
    for ax, image, title, cmap in zip(axs.ravel(), images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    pyplot.tight_layout()
    if path is not None:
        pyplot.savefig(path, dpi=300)
    else:
        pyplot.show()

def display_grid(images:list, titles:list, gridsize:tuple, figsize:tuple=(15, 15), suptitle:str=None, cmap:str='gray', path:str=None, dpi:int=300) -> None:
    '''Displays multiple images'''
    fig, axs = pyplot.subplots(nrows=gridsize[0], ncols=gridsize[1], figsize=figsize)
    for ax, image, title in zip(axs.ravel(), images, titles):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
        pyplot.tight_layout(pad=2)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1.05, fontsize=30)
    if path is not None:
        pyplot.savefig(path, dpi=dpi)
    else:
        pyplot.show()
    
def display_history(history:dict, metrics:list) -> None:
    '''Displays model training history'''
    fig, axs = pyplot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax, stat in zip(axs.ravel(), metrics):
        ax.plot(history[stat])
        ax.plot(history[f'val_{stat}'])
        ax.set_title(f'Training {stat}', fontsize=15)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training sample', 'Validation sample'], frameon=False)
    pyplot.tight_layout()
    pyplot.show()

def display_structure(model, filepath:str, type:str) -> None:
    '''Displays keras model structure'''
    summary = pd.DataFrame([dict(Name=layer.name, Type=layer.__class__.__name__, Shape=layer.output_shape, Params=layer.count_params()) for layer in model.layers])
    if type=='table': summary.style.to_html(filepath, index=False) 
    if type=='graph': utils.plot_model(model, to_file=filepath, show_shapes=True)