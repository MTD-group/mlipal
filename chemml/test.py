from chemml.optimization.active import ActiveLearning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob, math, re, os, random

from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, ensemble, tree, neighbors
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.layers import Concatenate, concatenate, Add, add
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

from machine_learning import RunML, RunDL, RunAL
from machine_learning import create_df_train, create_df_test, generate_descriptor

if __name__ == "__main__":

    test_num = 250
    total_train = 10000
    poly_train = 2000
    ev_train = 200
    random_train = total_train - (2*poly_train + ev_train)

    df = pd.read_csv('new_descriptor.csv')

    df_test = df[df['type'] == 'ev'].sample(n = test_num, random_state=1234567)
    df_test_index = df_test.index
    df_edit = df.drop(df_test_index)

    df_ev = df_edit[df_edit['type'] == 'ev'].sample(n = ev_train, random_state=1234567)
    df_poly_small = df_edit[df_edit['type'] == 'poly_small'].sample(n = poly_train, random_state=1234567)
    df_poly_large = df_edit[df_edit['type'] == 'poly_large'].sample(n = poly_train, random_state=1234567)
    df_random = df_edit[df_edit['type'] == 'random'].sample(n = random_train, random_state=1234567)
    df_train = pd.concat([df_ev, df_poly_small, df_poly_large, df_random])

    df_train_index = df_train.index

    dataproperty = 'energy'
    remove_column = ['hash', 'type', 'element', 'energy', 'force0', 'force1', 'force2']

    print('done')

    #result = RunDL(df_train, df_test, dataproperty, remove_column)

    out_dir = 'AL_testcase' # Output directory for AL
    layer = 'dense32' # Layer to extract info for AL
    al_component = [1, 50, 50, 50] #[loop, train_size, test_size, batch_size]

    # Perform active learning analysis
    df_train_al = df.iloc[df_train_index, :]
    df_test_al = df.iloc[df_test_index, :]
    df_all = pd.concat([df_test_al, df_train_al]).reset_index()
    df_all = df_all.drop('index', axis=1)
    RunAL(df_all, dataproperty, remove_column, out_dir, layer, al_component)
