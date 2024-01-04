#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import concurrent.futures
import geopandas as gpd
import cv2
import numpy as np
import os

from landuse1860_utilities import *
from skimage import color, exposure

# Utilities
classes = dict(zip(['undefined', 'buildings', 'transports', 'crops', 'meadows', 'pastures', 'specialised', 'forests', 'water', 'border'], np.arange(10)))

#%% COMPUTES LABELS

def compute_label(srcfile:str, dstfile:str) -> None:
    if os.path.exists(dstfile):
        return
    print(f'Processing: {filename(srcfile)}')
    y_landuse   = rasterise(landuse,   profile=srcfile, varname='y')
    y_border    = rasterise(border,    profile=srcfile, varname='y')
    y_river     = rasterise(river,     profile=srcfile, varname='y')
    y_transport = rasterise(transport, profile=srcfile, varname='y')
    y_building  = rasterise(building,  profile=srcfile, varname='y')
    label = np.copy(y_landuse)
    label = np.where(y_border    !=0, y_border,    label)
    label = np.where(y_river     !=0, y_river,     label)
    label = np.where(y_transport !=0, y_transport, label)
    label = np.where(y_building  !=0, y_building,  label)
    write_raster(label, srcfile, dstfile)

# Loads tiles and classes
classes = pd.read_csv(f"{paths['utilities']}/cartoem_classes.csv", usecols=['cartoem_id', 'y', 'label'])
classes = classes.sort_values(by=['y', 'cartoem_id'])

# Loads landuse vectors
landuse = gpd.read_file(f"{paths['vectors']}/cartoem/ocs_ancien_sans_bati.gpkg")[['THEME', 'geometry']]
landuse = landuse.merge(classes, how='left', left_on='THEME', right_on='cartoem_id')[['y', 'geometry']]

# Loads administrative boundaries
border = gpd.read_file(f"{paths['vectors']}/cartoem/limite_administrative.gpkg").geometry
border = gpd.GeoDataFrame({'y':9, 'geometry':border})

# Loads river vectors
river = gpd.read_file(f"{paths['vectors']}/cartoem/troncon_de_cours_d_eau.gpkg").geometry
river = gpd.GeoDataFrame({'y':8, 'geometry':river})
river.geometry = river.geometry.buffer(4)

# Loads transport vectors
transport = pd.concat((
    gpd.read_file(f"{paths['vectors']}/cartoem/troncon_de_route.gpkg").geometry,
    gpd.read_file(f"{paths['vectors']}/cartoem/troncon_de_voie_ferree.gpkg").geometry
))
transport = gpd.GeoDataFrame({'y':2, 'geometry':transport})
transport.geometry = transport.geometry.buffer(4)

# Loads building vectors
building = gpd.read_file(f"{paths['vectors']}/cartoem/batiment.gpkg")[['THEME', 'geometry']]
building = building.merge(classes, how='left', left_on='THEME', right_on='cartoem_id')[['y', 'geometry']]

# Computes labels
tiles    = gpd.read_file(f"{paths['vectors']}/tiles/cartoem_tiles.gpkg")
srcfiles = [f"{paths['images']}/image_{tile}.tif" for tile in tiles.tile]
dstfiles = [f"{paths['labels']}/label_{tile}.tif" for tile in tiles.tile]

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(compute_label, srcfiles, dstfiles)

del classes, landuse, river, transport, building, tiles, srcfiles, dstfiles

# Removes empty rasters
srcfiles = search_data(paths['labels'], 'label_.*.tif')

for srcfile in srcfiles:
    print(f'Processing: {filename(srcfile)}')
    label = read_raster(srcfile)
    if np.count_nonzero(label) / np.prod(label.shape) < 0.1 :
        os.remove(srcfile)

del srcfiles, srcfile, label

#%% PREPROCESSES IMAGES

def preprocess_image(srcfile, dstfile):
    image = read_raster(srcfile)
    image = color.rgb2lab(image)
    image = exposure.rescale_intensity(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    image[...,0] = clahe.apply(image[...,0])
    write_raster(image, srcfile, dstfile)

# srcfiles = search_data(paths['images'], 'image_.*.tif')
dstfiles = [f"{paths['images']}/image_{filename(srcfile)}.tif" for srcfile in srcfiles]

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(preprocess_image, srcfiles, dstfiles)