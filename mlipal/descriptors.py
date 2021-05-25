import amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
import numpy as np

def two_body_gaussians(atoms, cutoff=3, num_etas=2, num_offsets=5, etas=None,
        offsets=None):
    ''' Generates two-body symmetry functions by spacing out gaussians and
    biasing sampling towards the equilibrium position. '''

    if type(atoms) is list:
        elements = atoms
    else:
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

    if type(atoms) is list:
        elements = atoms
    else:
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

def main():
    ''' This currently doesn't do anything but I'm keeping it as reference for
    our current descriptor set. '''

    # Sets descriptor set
    cutoff = 3

    elements = ['Si']
    symm_funcs = make_symmetry_functions(elements=elements, type='G2',
            etas=[1, 20], offsets=np.array([0, 0.3, 0.5, 0.68, 0.8])*cutoff)
    descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)
