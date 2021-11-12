import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pymatgen.io.vasp import Poscar
import amp
import ase
import glob, re
#from afps import AmpFingerPrintSeeker
from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.utilities import Logger, make_filename, hash_images, get_hash
from ase import io
from ase.calculators.kim import KIM
from ase.build import bulk
from ase.db import connect

from machine_learning import RunML, RunDL, RunAL
from machine_learning import create_df_train, create_df_test, generate_descriptor


kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'

def two_body_gaussians(atoms, cutoff=3, num_etas=2, num_offsets=5, etas=None,
        offsets=None):
    ''' Generates two-body symmetry functions by spacing out gaussians and
    biasing sampling towards the equilibrium position. '''

    elements = list(set(atoms.get_chemical_symbols()))
    
    # TODO: take num_etas and num_offsets and systematically generate etas and
    # offsets arrays, only if etas and offsets are None. For now, resorts to
    # default
    if etas == None:
        etas = [0.1, 1, 20]
        # Generate etas
    if offsets == None:
        offsets = np.array([0, 0.3, 0.5, 0.68, 0.72, 0.8, 0.9]) * cutoff
        # Generate offsets

    symm_funcs = make_symmetry_functions(elements=elements, type='G2',
            etas=etas, offsets=offsets)
    #descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

    return symm_funcs

def three_body_gaussians(atoms, num_etas=2, num_zetas=2, num_gammas=2,
        etas=None, zetas=None, gammas=None, **kwargs):
    ''' Generates three-body symmetry functions by spacing out gaussians and
    biasing sampling towards the equilibrium positions. '''

    elements = list(set(atoms.get_chemical_symbols()))
    
    # TODO: systematically generate etas and
    # offsets arrays, only if parameters (etas, zetas, gammas) are None. For
    # now, resorts to default
    if etas is None:
        etas = [0.01,1]
        # Generate etas
    if zetas is None:
        zetas = [0.01, 0.1, 0.5, 1]
        # Generate offsets
    if gammas is None:
        gammas = [-1, 1]
        # Generate offsets

    # NOTE: once we switch to using DFT, change to G4
    symm_funcs = make_symmetry_functions(elements=elements, type='G5',
            etas=etas, zetas=zetas, gammas=gammas, **kwargs)
    #descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

    return symm_funcs

def df_database(db):
    count = 0
    atom_data = []

    for atoms in db.select():
        symm_func = two_body_gaussians(atoms.toatoms()) + three_body_gaussians(atoms.toatoms())
        descriptor = Gaussian(Gs=symm_func, cutoff=3)
        descriptor.calculate_fingerprints({atoms.hash: atoms.toatoms()})

        atoms_hash = get_hash(atoms.toatoms())        
        hashes = np.asarray(atoms_hash).reshape(1,)

        atoms_type = atoms.type
        atoms_type = np.asarray(atoms_type).reshape(1,)

        energy = atoms.energy / len(descriptor.fingerprints[atoms_hash])
        energy = np.array(energy).reshape(1,)

        forces = atoms.forces

        for i in range(len(descriptor.fingerprints[atoms_hash])):
            atoms_fp = descriptor.fingerprints[atoms_hash][i]
            element = np.asarray(atoms_fp[0]).reshape(1,)
            feature = np.array(atoms_fp[1])
            force = forces[i]
            atom_info = np.concatenate([hashes, atoms_type, element, energy, force, feature])
            atom_info = np.reshape(atom_info, (1, atom_info.shape[0]))
            if count==0:
                atom_data = atom_info
                count = 1
            else:
                atom_data = np.append(atom_data, atom_info, axis=0)     

    df = pd.DataFrame(atom_data)  
    df = df.rename(columns={0: "hash", 1:"type", 2: "element", 3: "energy", 4: "force0", 5: "force1", 6: "force2"})             

    return df



if __name__ == "__main__":

    db = connect('structures.db')

    df = df_database(db)
    
    df.to_csv('new_contour_descriptor.csv', index=False) 
