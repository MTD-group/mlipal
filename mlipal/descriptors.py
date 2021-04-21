import amp
from amp import Amp 
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
import ase
import numpy as np
import os
import re

def calculate_fingerprints(atoms, descriptor):
    ''' Calculates fingerprints of an ASE Atoms object using the given
    descriptor set.

    Inputs:
        atoms: ASE Atoms object. This function can (currently) only take one
            Atoms object as input.
        descriptor: AMP descriptor object. Descriptor set which will be used to
            compute the fingerprints.

    Outputs:
        fingerprints: dict. {hash: fingerprints} '''

    atoms_hash = amp.utilities.get_hash(atoms)

    descriptor.calculate_fingerprints({atoms_hash: atoms})

    # NOTE: should this return just the fingerprints, without the hash?
    return {atoms_hash: descriptor.fingerprints[atoms_hash]}

def two_body_gaussians(atoms, cutoff=3, num_etas=2, num_offsets=5, etas=None,
        offsets=None):
    ''' Generates two-body descriptors by spacing out gaussians and biasing
    sampling towards the equilibrium position. '''

    elements = set(atoms.get_chemical_symbols())
    
    cutoff = 3
    # TODO: take num_etas and num_offsets and systematically generate etas and
    # offsets arrays, only if etas and offsets are None. For now, resorts to
    # default
    if etas == None:
        etas = [1,20]
        # Generate etas
    if offsets == None:
        offsets = np.array([0, 0.3, 0.5, 0.68, 0.8]) * cutoff
        # Generate offsets

    symm_funcs = make_symmetry_functions(elements=elements, type='G2',
            etas=etas, offsets=offsets)
    descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

    return descriptor

def main():
    ''' This currently doesn't do anything but I'm keeping it as reference for
    our current descriptor set. '''

    # Sets descriptor set
    cutoff = 3

    elements = ['Si']
    symm_funcs = make_symmetry_functions(elements=elements, type='G2',
            etas=[1, 20], offsets=np.array([0, 0.3, 0.5, 0.68, 0.8])*cutoff)
    descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)
