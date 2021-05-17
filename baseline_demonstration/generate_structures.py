import amp.utilities
import ase
from ase.calculators.kim import KIM
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
    v1 = np.linspace(0.5, 0.625, np.ceil(nev / 5))
    v2 = np.linspace(0.625, 1.625, np.ceil(3 * nev / 5))
    v3 = np.linspace(1.625, 2.4, np.ceil(nev/5))
    ev_structures = gen.volume_series.volume_series(vol_mults = np.concatenate((v1,v2,v3)))

    with ase.db.connect(db_filename) as db:
        for atoms in ev_structures:
            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash, type='ev')

    # PolymorphD3
