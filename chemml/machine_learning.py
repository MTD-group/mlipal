import warnings
warnings.filterwarnings('ignore')

from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.utilities import Logger, make_filename, hash_images, get_hash
from ase.calculators.kim import KIM
from ase import io
#from ase.calculators.kim import KIM

#from pymatgen.io.vasp import Poscar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob, math, re, os, random

from sklearn.model_selection import train_test_split
#from matminer.featurizers.base import MultipleFeaturizer
#from matminer.featurizers.conversions import DictToObject
#from matminer.featurizers.structure import (DensityFeatures, MaximumPackingEfficiency, 
#                                            SiteStatsFingerprint, ChemicalOrdering, 
#                                           StructuralHeterogeneity,RadialDistributionFunction,
#                                           DensityFeatures) 


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

from chemml.optimization import ActiveLearning

descriptor_size = 37

def generate_descriptor():

    kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'
    #my_model_calc = KIM(kim_calc)
    my_model_calc = None

    # Generates descriptors
    cutoff = 3
    elements = ['Si']
    g2_etas = [1, 20]
    g2_offsets = offsets=np.array([0, 0.3, 0.5, 0.68, 0.8])*cutoff

    g2_symm_funcs = make_symmetry_functions(elements=elements, type='G2', etas=g2_etas,
        offsets=g2_offsets)

    symm_funcs = g2_symm_funcs
    descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

    #model = NeuralNetwork()

    #log_file = 'amp_d_calc.log'
    #MLIP = Amp(descriptor = descriptor, model=model)
    #MLIP.cores['localhost'] = cores = 1
    #MLIP._log = Logger(make_filename('',log_file))

    return descriptor


def create_df_train(poscarfilepaths, forcefilepaths, descriptor):
    count = 0
    for poscarfile, forcefile in zip(poscarfilepaths, forcefilepaths):
        atoms = io.read(poscarfile)
        atoms_hash = get_hash(atoms)
        descriptor.calculate_fingerprints({atoms_hash: atoms})
        atoms_fp = descriptor.fingerprints[atoms_hash][0]
        element = np.asarray(atoms_fp[0]).reshape(1,)
        feature = np.array(atoms_fp[1])

        allforces = open(forcefile, 'r') 
        forces = allforces.readlines()
        force = np.asarray(forces).astype(str)[0]
        energy = float(re.search('Energy:\t(.*)\n', force).group(1))
        energy = np.array(energy).reshape(1,)

        atom_info = np.concatenate([energy, element, feature])
        atom_info = np.reshape(atom_info, (1, atom_info.shape[0]))

        if count!=0:
            atom_data = np.append(atom_data, atom_info, axis=0)
        else:
            atom_data = atom_info
            count = 1

    df = pd.DataFrame(atom_data)  
    df = df.rename(columns={0: "forces", 1: "element"})

    return df

def create_df_test(descriptor, data):
    for i in range(data.shape[0]):
        #atom_key = (data[i][0], data[i][1], i)
        atom_key = (data[i][0], data[i][1])
        atom_key = np.asarray(atom_key).reshape(1,len(atom_key))
        atom_type = np.asarray(re.findall(r"'(.*?)'", str(data[i][2]), re.DOTALL))
        atom_type = np.reshape(atom_type, (atom_type.shape[0], 1))
        atom_val = re.findall(r", \[(.*?)\]\)", str(data[i][2]), re.DOTALL)
        atom_store = np.empty(shape=(len(atom_val), descriptor))
    
        atom_key[0][1] = (float(atom_key[0][1])/len(atom_val))
        atom_key = np.repeat(atom_key, len(atom_val), axis=0)
        for j in range(len(atom_val)):
            atom_value = np.asarray(atom_val[j].split(','))
            atom_store[j] = atom_value
        atom_info = np.concatenate([atom_key, atom_type, atom_store], axis=1)
        if i!=0:
            atom_data = np.append(atom_data, atom_info, axis=0)
        else:
            atom_data = atom_info
        
    # atom_data.shape        
    # len(np.unique(atom_data[:, 1]))
    df = pd.DataFrame(atom_data)
    #df = df.rename(columns={0: "hash", 1: "forces", 2: "group", 3: "element"})
    df = df.rename(columns={0: "hash", 1: "forces", 2: "element"})
    
    indexes = pd.DataFrame(df.groupby(['hash']).count()['forces'])
    indexes.reset_index(inplace=True)
    index = np.asarray(indexes)
    
    for i in range(len(index)):
        for j in range(len(df)):
            if index[i][0] == df['hash'][j]:
                df['forces'][j] = float(df['forces'][j])/index[i][1]

    df = df.drop(columns=['hash'])

    return df


def RunML(train, test, dataproperty, remove_column):    
    train_elem = train.drop(remove_column, axis=1)
    test_elem = test.drop(remove_column, axis=1)
    
    new_x_train = np.asarray(train_elem.values, dtype=np.float)
    new_x_test = np.asarray(test_elem.values, dtype=np.float)
    
    new_y_train = np.array(train[dataproperty])
    new_y_test = np.array(test[dataproperty])

    new_y_train.shape = (len(new_y_train),)
    new_y_test.shape = (len(new_y_test),)
    
    # AdaBoost
    ad = ensemble.AdaBoostRegressor(random_state=0).fit(new_x_train, new_y_train)
    y_pred_ad = ad.predict(new_x_test)
    MAE_AD = mean_absolute_error(new_y_test, y_pred_ad)
    print("Test MAE for AdaBoost is {}".format(MAE_AD))
    
    #ElasticNet
    en = linear_model.ElasticNet(random_state=0).fit(new_x_train, new_y_train)
    y_pred_en = en.predict(new_x_test)
    MAE_EN = mean_absolute_error(new_y_test, y_pred_en)
    print("Test MAE for ElasticNet is {}".format(MAE_EN))

    
    # Linear Regression
    lr = linear_model.LinearRegression().fit(new_x_train, new_y_train)
    y_pred_lr = lr.predict(new_x_test)
    MAE_LR = mean_absolute_error(new_y_test, y_pred_lr)
    print("Test MAE for Linear Regression is {}".format(MAE_LR))
    
    # SGD Regression
    sgdr = linear_model.SGDRegressor(penalty='l1', l1_ratio=0.01).fit(new_x_train, new_y_train)
    y_pred_sgdr = sgdr.predict(new_x_test)
    MAE_SGDR1 = mean_absolute_error(new_y_test, y_pred_sgdr)
    sgdr = linear_model.SGDRegressor(penalty='l2', l1_ratio=0.01).fit(new_x_train, new_y_train)
    y_pred_sgdr = sgdr.predict(new_x_test)
    MAE_SGDR2 = mean_absolute_error(new_y_test, y_pred_sgdr)
    if MAE_SGDR1 > MAE_SGDR2:
        print("Test MAE for SGD Regression is {}".format(MAE_SGDR2))
    else:
        print("Test MAE for SGD Regression is {}".format(MAE_SGDR1))

    # Ridge
    r = linear_model.Ridge(alpha=0.001, fit_intercept=False, normalize=True, solver='svd', tol=0.0001, random_state=0).fit(new_x_train, new_y_train)
    y_pred_r = r.predict(new_x_test)
    MAE_R1 = mean_absolute_error(new_y_test, y_pred_r)
    r = linear_model.Ridge(alpha=0.001, fit_intercept=True, normalize=True, solver='svd', tol=0.0001, random_state=0).fit(new_x_train, new_y_train)
    y_pred_r = r.predict(new_x_test)
    MAE_R2 = mean_absolute_error(new_y_test, y_pred_r)
    if MAE_R1 > MAE_R2:
        print("Test MAE for Ridge is {}".format(MAE_R2))
    else:
        print("Test MAE for Ridge is {}".format(MAE_R1))
    
    # SVR
    svrin = [1, 10, 100, 1000]
    MAE_RBFSVM = 1000000000000000
    for i in range(len(svrin)):
        rbfsvm = svm.SVR(C=svrin[i], gamma='scale').fit(new_x_train, new_y_train)
        y_pred_rbfsvm = rbfsvm.predict(new_x_test)
        temp = mean_absolute_error(new_y_test, y_pred_rbfsvm)
        if temp < MAE_RBFSVM:
            MAE_RBFSVM = temp
    print("Test MAE for SVR is {}".format(MAE_RBFSVM))
    
    # Desicion Tree
    dtin = [2,5,10,100,1000]
    MAE_DT = 1000000000000000
    for i in range(len(dtin)):
        dt = tree.DecisionTreeRegressor(max_depth=dtin[i], random_state=0).fit(new_x_train, new_y_train)
        y_pred_dt = dt.predict(new_x_test)
        temp = mean_absolute_error(new_y_test, y_pred_dt)
        if temp < MAE_DT:
            MAE_DT = temp
    print("Test MAE for Desicion Tree is {}".format(MAE_DT))
    
    #ExtraTree
    et = tree.ExtraTreeRegressor(max_depth=25, random_state=0).fit(new_x_train, new_y_train)
    y_pred_et = et.predict(new_x_test)
    MAE_ET = mean_absolute_error(new_y_test, y_pred_et)
    print("Test MAE for Extra Tree is {}".format(MAE_ET))
    
    #Bagging
    b = ensemble.BaggingRegressor(random_state=0).fit(new_x_train, new_y_train)
    y_pred_b = b.predict(new_x_test)
    MAE_B = mean_absolute_error(new_y_test, y_pred_b)
    print("Test MAE for Bagging is {}".format(MAE_B))
    
    # Random Forest
    rfin = [100, 150, 200]
    rfin2 = [5, 10, 15, 20]
    MAE_RF = 1000000000000000
    for i in range(len(rfin)):
        for j in range(len(rfin2)):
            rf = ensemble.RandomForestRegressor(max_depth=25, n_estimators=rfin[i], min_samples_split=rfin2[j], random_state=0).fit(new_x_train, new_y_train)
            y_pred_rf = rf.predict(new_x_test)
            temp = mean_absolute_error(new_y_test, y_pred_rf)
            if temp < MAE_RF:
                MAE_RF = temp
    print("Test MAE for Random Forest is {}".format(MAE_RF))
    
    #Lasso
    lain = [10, 1, 0.1, 0.01]
    MAE_LA = 1000000000000000
    for i in range(len(lain)):
        la = linear_model.Lasso(alpha=0.01, random_state=0).fit(new_x_train, new_y_train)
        y_pred_la = la.predict(new_x_test)
        temp = mean_absolute_error(new_y_test, y_pred_la)
        if temp < MAE_LA:
            MAE_LA = temp
    print("Test MAE for Lasso is {}".format(MAE_LA))


def create_model():

    model = Sequential()
    model.add(Dense(1024, name='dense1024', activation='relu', input_dim=descriptor_size))
    model.add(Dense(512, name='dense512', activation='relu'))
    model.add(Dense(256, name='dense256', activation='relu'))
    model.add(Dense(128, name='dense128', activation='relu'))
    model.add(Dense(64, name='dense64', activation='relu'))
    model.add(Dense(32, name='dense32', activation='relu'))
    model.add(Dense(1, activation='linear'))

    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    
    return model


def RunDL(train, test, dataproperty, remove_column):
    training, val = train_test_split(train, test_size=0.1, random_state=1234567)
    
    train_elem = training.drop(remove_column, axis=1)
    val_elem = val.drop(remove_column, axis=1)
    test_elem = test.drop(remove_column, axis=1)
    
    new_x_train = np.asarray(train_elem.values, dtype=np.float)
    new_x_val = np.asarray(val_elem.values, dtype=np.float)
    new_x_test = np.asarray(test_elem.values, dtype=np.float)

    new_y_train = np.array(training[dataproperty])
    new_y_val = np.array(val[dataproperty])
    new_y_test = np.array(test[dataproperty])

    new_y_train.shape = (len(new_y_train),)
    new_y_val.shape = (len(new_y_val),)
    new_y_test.shape = (len(new_y_test),)
    
    model = create_model()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
    model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_val, new_y_val), epochs=200, batch_size=32, callbacks=[es])
    result = model.evaluate(new_x_test, new_y_test, batch_size=32)

    '''
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    '''

    return result    

def compute_energy(atoms, calc):
    atoms.set_calculator(calc)
    
    return atoms.get_potential_energy()

def RunAL(dataset, dataproperty, remove_column, out_dir, layer, al_component):
    ''' Runs active learning loop.

    dataset: pandas DataFrame
        Contains training and test data.
    dataproperty: str
        Column name in dataset corresponding to the property of interest (i.e.
        energy)
    remove_column: list of strings
        Column names to drop from dataset when performing training. Should drop
        everything except for the feature vector?
    out_dir: str
        Directory to put output files
    layer: str
        Layer to extract information for active learning
    al_component: list of ints
        Parameters for active learning: [# of loops, training size, test size,
        batch size]
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset.to_csv(out_dir+'/dataset.csv', index=False)    
    
    df_all = dataset.drop(remove_column, axis=1)
    df_forces = dataset[dataproperty]

    # all data sets must be 2-dimensional
    y = df_forces.values.reshape(-1,1)

    al = ActiveLearning(
           model_creator = create_model,
           U = df_all,
           target_layer = [layer], 
           train_size = al_component[1],  # tr_size initial training data will be selected randomly
           test_size = al_component[2],  # te_size independent test data will be selected randomly for the entire search
           batch_size = [al_component[3],0,0] # at each round of AL, labels for ba_size candidates will be queried using EMC method
           )

    tr_ind, te_ind = al.initialize(random_state=1234567)


    al.deposit(tr_ind, y[tr_ind])
    al.deposit(te_ind, y[te_ind])

    # OpenKIM Tersoff calculator to compute energies
    kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'
    calc = KIM(kim_calc)

    while al.query_number < al_component[0]:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=50, verbose=0, mode='auto')
        tr_ind = al.search(n_evaluation=3, ensemble='kfold', n_ensemble=3,
                normalize_input=True, normalize_internal=False, batch_size=32,
                epochs=500, verbose=0,
                callbacks=[early_stopping],validation_split=0.1)

        print(tr_ind)

        al.results.to_csv(out_dir+'/emc.csv',index=False)

        pd.DataFrame(al.train_indices).to_csv(out_dir+'/train_indices.csv',index=False)
        pd.DataFrame(al.test_indices).to_csv(out_dir+'/test_indices.csv',index=False)
        pd.DataFrame(tr_ind).to_csv(out_dir+'/next_indices.csv',index=False)


        tr_energies = []
        for ind in tr_ind:
            hash_ = dataset.iloc[ind]['hash']
            atoms = db.get_atoms(hash==hash_)
            tr_energies.append(compute_energy(atoms, calc))

        # XXX is there a way to do this through a list comprehension?
        #tr_energies = [compute_energy(row.toatoms(), calc) for row in
        #        db.select()]
        al.deposit(tr_ind, tr_energies)

        # you can run random search if you want to
        #al.random_search(y, n_evaluation=3, random_state=13, batch_size=32, epochs=500, verbose=0, callbacks=[early_stopping],validation_split=0.05)

        #al.random_results.to_csv(out_dir+'/random.csv',index=False)

        plots = al.visualize(y)
        if not os.path.exists(out_dir+"/plots"):
            os.makedirs(out_dir+"/plots")

        plots['dist_pc'][0].savefig(out_dir+"/plots/dist_pc_0_%i.png"%al.query_number , close = True, verbose = True)
        plots['dist_y'][0].savefig(out_dir+"/plots/dist_y_0_%i.png"%al.query_number , close = True, verbose = True)
        plots['learning_curve'].savefig(out_dir+"/plots/lcurve_%i.png"%al.query_number, close = True, verbose = True)    

def insert_row(df, my_row):
    df.loc[len(df)] = my_row


"""
if __name__ == "__main__":

    descriptor = generate_descriptor()

    folderpath = '../../poscar/'
    poscarfilepaths = glob.glob(folderpath + '*.POSCAR')
    forcefilepaths = glob.glob(folderpath + '*.POSCAR.forces')

    df_train = create_df_train(poscarfilepaths, forcefilepaths)      

    #train = np.load("../../Si_random_fingerprints.npy", allow_pickle=True)
    test = np.load("../../Si_e_v_fingerprints.npy", allow_pickle=True)   

    #df_train = create_df_new(10, train)
    df_test = create_df_test(10, test)

    dataproperty = 'forces'
    #remove_column = ['hash', 'forces', 'element'] 
    remove_column = ['element', 'forces']

    #RunML(df_train, df_test, dataproperty, remove_column)
    #result = RunDL(df_train, df_test, dataproperty, remove_column)

    #out_dir = 'AL_case1'
    #layer = 'dense32'
    #al_component = [5, 30, 30, 30] #[loop, train_size, test_size, batch_size]
    #RunAL(df_train, df_test, dataproperty, remove_column, out_dir, layer, al_component)
"""
