#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes statistics for the Landuse project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import concurrent.futures as future
import itertools
import functools
import numpy as np
import pandas as pd

from skimage import filters
from sklearn import metrics
from landuse1860_utilities import *

# Utilities
train_sample = '(label|predict)_(0120_6800|0120_6860|0140_6780|0140_6800|0140_6820|0140_6840|0160_6780|0160_6820|0160_6840|0160_6880|0180_6780|0180_6800|0200_6780|0200_6820|0200_6840|0200_6860|0220_6720|0220_6760|0220_6780|0220_6820|0220_6840|0220_6860|0220_6880|0240_6740|0240_6760|0240_6800|0240_6820|0240_6840|0240_6860|0240_6880|0260_6740|0260_6760|0260_6780|0260_6800|0260_6820|0260_6840|0260_6860|0280_6740|0280_6780|0280_6800|0280_6820|0280_6840|0280_6860|0300_6740|0300_6760|0300_6780|0300_6800|0300_6820|0300_6840|0300_6860|0320_6820|0600_6680|0600_7060|0600_7100|0620_6440|0620_6620|0620_6640|0620_6660|0620_6680|0620_6700|0620_7020|0620_7040|0620_7060|0620_7080|0620_7100|0640_6400|0640_6420|0640_6440|0640_6460|0640_6540|0640_6600|0640_6620|0640_6660|0640_6680|0640_6700|0640_6720|0640_7020|0640_7040|0640_7060|0640_7120|0660_6420|0660_6440|0660_6460|0660_6480|0660_6520|0660_6560|0660_6580|0660_6620|0660_6660|0660_6700|0660_7020|0660_7060|0660_7080|0660_7100|0660_7120|0680_6420|0680_6440|0680_6460|0680_6480|0680_6500|0680_6520|0680_6540|0680_6560|0680_6580|0680_6600|0680_6620|0680_6640|0680_6660|0680_6700|0680_6720|0680_7060|0700_6400|0700_6440|0700_6460|0700_6480|0700_6540|0700_6560|0700_6580|0700_6600|0700_6620|0700_6940|0700_6960|0700_7000|0700_7020|0700_7040|0700_7060|0700_7080|0720_6380|0720_6400|0720_6460|0720_6480|0720_6500|0720_6520|0720_6540|0720_6580|0720_6600|0720_6620|0720_6900|0720_6920|0720_6940|0720_6960|0720_7020|0720_7040|0720_7060|0740_6340|0740_6360|0740_6380|0740_6400|0740_6420|0740_6440|0740_6460|0740_6480|0740_6500|0740_6520|0740_6540|0740_6600|0740_6620|0740_6640|0740_6880|0740_6920|0740_6940|0740_6960|0740_7000|0740_7040|0760_6380|0760_6420|0760_6440|0760_6500|0760_6520|0760_6540|0760_6560|0760_6580|0760_6600|0760_6940|0760_6960|0760_6980|0760_7000|0760_7040|0780_6360|0780_6380|0780_6420|0780_6440|0780_6460|0780_6480|0780_6500|0780_6520|0780_7000|0780_7020|0800_6280|0800_6380|0800_6400|0800_6420|0800_6440|0800_6480|0800_6520|0800_6540|0800_6580|0820_6260|0820_6320|0820_6400|0820_6480|0840_6260|0840_6300|0840_6320|0840_6440|0840_6460|0860_6280|0860_6300|0880_6260|0880_6280|0880_6300|0900_6260|0900_6280|0900_6300|0920_6540|0940_6580|0940_6600|0960_6540|0960_6560|0960_6580|0960_6600|0980_6560|0980_6600|1000_6560).tif$'
valid_sample = '(label|predict)_(0120_6840|0160_6800|0180_6820|0180_6840|0200_6760|0200_6800|0220_6800|0320_6840|0600_7040|0620_6460|0640_6640|0640_6740|0640_7080|0660_6540|0660_6600|0660_6720|0680_7020|0680_7040|0700_6360|0700_6500|0700_6520|0700_6640|0700_6980|0720_6360|0720_6440|0720_6880|0720_7000|0740_6900|0740_6980|0760_6360|0760_6480|0780_6400|0780_6560|0780_6580|0800_6500|0820_6460|0840_6280|0900_6240|0940_6560|0980_6540|0980_6580).tif$'
test_sample  = '(label|predict)_(0140_6860|0140_6880|0160_6860|0180_6860|0180_6880|0200_6880|0240_6780|0280_6760|0600_7080|0620_6420|0640_6480|0640_7100|0660_6500|0660_6640|0660_6680|0660_7040|0680_6680|0680_7000|0680_7080|0700_6380|0700_6420|0700_6660|0700_6900|0700_6920|0720_6420|0720_6560|0720_6980|0740_6560|0740_6580|0740_7020|0760_6400|0760_6460|0760_7020|0780_6540|0780_6980|0800_6360|0800_6460|0820_6280|0820_6300|0820_6380|0820_6420|0820_6440|0820_6500|0860_6260|0920_6560|0920_6580|0940_6520|0940_6540).tif$'

#%% FUCNTIONS

def compute_confmat(label:np.ndarray, predictions:np.ndarray, mask:np.ndarray, dropclasses:list) -> pd.DataFrame:
    '''Computes cinfurion matrix'''
    confmat = pd.DataFrame({'y' : label[mask], 'yh' : predictions[mask]})
    confmat = pd.crosstab(confmat.y, confmat.yh, margins=False)
    confmat = confmat.reindex(index=classes.values(), columns=classes.values(), fill_value=0)
    confmat = confmat.drop(dropclasses, axis=0)
    confmat = confmat.drop(dropclasses, axis=1)
    confmat = confmat.rename_axis('True', axis=0)
    confmat = confmat.rename_axis('Predicted', axis=1)
    confmat = confmat.rename(dict(zip(classes.values(), classes.keys())), axis=0)
    confmat = confmat.rename(dict(zip(classes.values(), classes.keys())), axis=1)
    return confmat

def make_binary(confmat:pd.DataFrame, target:str) -> dict:
    tp = confmat.loc[target, target]
    tn = confmat.drop(index=target, columns=target).values.sum()
    fp = confmat.loc[:, target].drop(target).values.sum()
    fn = confmat.loc[target, :].drop(target).values.sum()
    stats = {'tp': tp, 'tn': tn, 'fp':fp, 'fn':fn}
    return stats

# Computes statistics from binary confusion matrix
def compute_statistics(confmat:dict, target:str):
    tp, tn, fp, fn = make_binary(confmat, target).values()
    # nobs       = fp + fn + tp
    recall     = tp / (tp + fn) # Number of correctly predicted positive observations / total of correctly predicted observations
    precision  = tp / (tp + fp) # Number of correctly predicted positive observations / total of positive observations
    fscore     = 2 * (precision * recall) / (precision + recall)
    statistics = dict(target=target, recall=recall, precision=precision, fscore=fscore)
    return statistics

#%% COMPUTES PREDICITON STATISTICS

# Paths
labels    = search_data(paths['labels'], pattern=test_sample)
preds_cnn = search_data(paths['postprocessed'], pattern=test_sample)
preds_rf  = search_data(f"{paths['data']}/predictions_rf", pattern=test_sample)

# Computes confusion matrices
def confmat_wrapper(label:str, pred:str) -> pd.DataFrame:
    label = read_raster(label)
    pred  = read_raster(pred)
    mask  = ~np.isin(label, [0, 9])
    # mask  = np.logical_and(~np.isin(label, [0, 9]), filters.sobel(label) == 0)
    confmat = compute_confmat(label, pred, mask, dropclasses=[0, 9])
    return confmat

with future.ThreadPoolExecutor(max_workers=6) as executor:
    confmats = list(executor.map(confmat_wrapper, labels, preds_rf))

confmats = functools.reduce(lambda cm1, cm2: cm1.add(cm2, fill_value=0), confmats)

# Generic
print(f'Accuracy: {accuracy:.04f}')

# Confusion matrix - True
print('Read as rows:\nAmong the pixels of [true], [value]% are predicted as [predicted]')
stat = confmats.divide(confmats.sum(axis=1), axis=0)

# Confusion matrix - Predicted
print('Read as columns:\nAmong the pixels predicted as [predicted], [value]% are in fact [true]')
stat = confmats.divide(confmats.sum(axis=0), axis=1)

stat.loc['Total', :] = stat.sum(axis=0)
stat.loc[:, 'Total'] = stat.sum(axis=1)
stat = stat.applymap(lambda value: '{0:.1f} %'.format(value * 100))
print(stat)

# Class- wise precision recall
targets    = ['buildings', 'crops', 'meadows', 'pastures', 'specialised', 'forests', 'water']
statistics = [compute_statistics(confmats, target) for target in targets]
statistics = pd.DataFrame(statistics).set_index('target')
statistics_rf  = statistics.round(2).copy()
statistics_cnn = statistics.round(2).copy()
