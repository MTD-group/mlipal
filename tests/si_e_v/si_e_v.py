import amp
import ase
from ase.calculators.kim import KIM
from ase.build import bulk
from mlipal.descriptors import three_body_gaussians
import numpy as np
import os
import sys

kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'

def compute_energy_and_forces(xtl, **kwargs):
    ''' Run static molecular mechanics calculation using KIM calculator (Tersoff
    potential T3). Since the energy and forces are written to the xtl Atoms
    object, this function does not need to return anything.
    
    Inputs:
        xtl: ASE Atoms object. Energy and forces are computed for this. '''

    calc = KIM(kim_calc)
    xtl.set_calculator(calc)

    xtl.get_potential_energy()
    xtl.get_forces()

def generate_volume_series(vol_mults = np.linspace(0.6, 2, 50)):
    ''' Generates crystalline Si structures in a volume series.
    Inputs:
        vol_mults: array-like. List of scaling factors. This scales the cell
            unit vectors, not the volume itself!
    Outputs:
        xtls: list. Contains Atoms objects with scaled volumes.'''

    # Equilibrium Si crystal structure
    xtl = bulk('Si', 'diamond', a=5.43)
    xtls = []
    for v in vol_mults:
        # Create copy of equilibrium crystal to scale by volume multiplier
        xtl_scaled = xtl.copy()
        xtl_scaled.set_cell(xtl_scaled.get_cell() * v, scale_atoms=True)

        xtls.append(xtl_scaled)

    return xtls

def calculate_si_e_v_fingerprints(vol_mults = np.linspace(0.6, 2, 50),
        descriptor=None):
    '''Example function which returns a volume series of scaled equilibrium Si
    structures along with energies and descriptors.'''

    # This sets up a series of lattice constant multipliers that is biased
    # towards the ground state lattice constant (i.e. v=1)
    v1 = np.linspace(0.6, 0.8, 81)
    v2 = np.linspace(0.8, 1.2, 401)
    v3 = np.linspace(1.2, 2, 201)
    vol_mults = np.concatenate( (v1, v2, v3) )

    xtls = generate_volume_series(vol_mults)
    images = amp.utilities.hash_images(xtls)

    descriptor = three_body_gaussians(xtls[0])
    descriptor.calculate_fingerprints(images)

    atoms_and_fingerprints = {}

    for xtl in xtls:
        atoms_hash = amp.utilities.get_hash(xtl)
        fp = descriptor.fingerprints[atoms_hash]

        compute_energy_and_forces(xtl)

        atoms_and_fingerprints[atoms_hash] = [xtl, fp]

    return atoms_and_fingerprints


def main():
    # This sets up a series of lattice constant multipliers that is biased
    # towards the ground state lattice constant (i.e. v=1)
    v1 = np.linspace(0.6, 0.8, 81)
    v2 = np.linspace(0.8, 1.2, 401)
    v3 = np.linspace(1.2, 2, 201)
    vol_mults = np.concatenate( (v1, v2, v3) )

    xtls = generate_volume_series(vol_mults)

    # Save each xtl w/ energy and forces computed to a trajectory file
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i, vol in enumerate(vol_mults):
        xtl = xtls[i]
        compute_energy_and_forces(xtl)
        xtl.write(os.path.join(data_dir, '{:.2f}.traj'.format(vol)))

if __name__ == '__main__':
    main()
