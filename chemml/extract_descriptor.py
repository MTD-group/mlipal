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

    test_size = 50
    train_size = 50
    batch_size = 50
    
    dataproperty = 'energy'
    remove_column = ['hash', 'type', 'element', 'energy', 'force0', 'force1', 'force2']


    df = pd.read_csv('../workflow_test/new_descriptor.csv')

    #result = RunDL(df_train, df_test, dataproperty, remove_column)

    out_dir = 'AL_testcase' # Output directory for AL
    layer = 'dense32' # Layer to extract info for AL
    al_component = [1, train_size, test_size, batch_size] #[loop, train_size, test_size, batch_size]

    # Perform active learning analysis
    RunAL(df, dataproperty, remove_column, out_dir, layer, al_component)
