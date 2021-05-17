import amp.utilities
import ase
from ase.calculators.kim import KIM
from ase.build import bulk
import ase.db
import mlipal.structure_generation as gen
import numpy as np

def main():
    ''' Generate training set '''

    # Number of each type of structure
    nev = 400
    npoly_small = 400 # small polymorphD3 distortions
    npoly_large = 400 # large polymorphD3 distortions
    nrandom = 800

    db_filename = 'structures.db'

    # E-V structures
    print('Generating EV structures')
    v1 = np.linspace(0.5, 0.625, np.ceil(nev / 5))
    v2 = np.linspace(0.625, 1.625, np.ceil(3 * nev / 5))
    v3 = np.linspace(1.625, 2.4, np.ceil(nev/5))
    ev_structures = gen.volume_series.volume_series(vol_mults = np.concatenate((v1,v2,v3)))

    with ase.db.connect(db_filename) as db:
        for atoms in ev_structures:
            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash, type='ev')
    print('Done!')

    # PolymorphD3
    equilibrium_si = bulk('Si', 'diamond', a=5.43)

    # Small distortions
    print('Generating small distortion polymorphs')
    params = {
            "atom_distortion": 0.1,
            "lattice_distortion": 0.05,
            "shrink_bias": 0.2,
            "deletion_chance": 0.02,
            "volume_change_max": 0.025
            }

    with ase.db.connect(db_filename) as db:
        for i in range(npoly_small):
            Poly = gen.polymorphD3.Poly(equilibrium_si, rcut=3, flip_chance=0,
                    swap_chance=0, **params)
            atoms = Poly.atoms_out

            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash, type='poly_small')
    print('Done!')

    # Large distortions
    print('Generating small distortion polymorphs')
    params = {
            "atom_distortion": 0.25,
            "lattice_distortion": 0.10,
            "shrink_bias": 0.25,
            "deletion_chance": 0.05,
            "volume_change_max": 0.05
            }

    with ase.db.connect(db_filename) as db:
        for i in range(npoly_large):
            Poly = gen.polymorphD3.Poly(equilibrium_si, rcut=3, flip_chance=0,
                    swap_chance=0, **params)
            atoms = Poly.atoms_out

            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash, type='poly_large')

    print('Done!')

    # Random structures
    print('Generating random structures')
    def composition(elements):
        # composition generator for atomic solid
        return np.array([1])

    with ase.db.connect(db_filename) as db:
        for i in range(nrandom):
            print('Random structure number:', i)
            atoms = gen.rrsm.reasonable_random_structure_maker('Si',
                    cut_off_radius = 3,
                    fill_factor_max = 0.65
                    fill_factor_min = 0.2,
                    composition_generator = composition)

            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash, type='random')

    print('Done!')
