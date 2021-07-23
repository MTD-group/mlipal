import amp.utilities
from ase.calculators.kim import KIM
import ase.db
from ase.md.contour_exploration import ContourExploration
from ase.optimize import BFGS
import numpy as np
import os

openkim_pot = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'

def main():
    maxstep = 1
    angle_limit = 30
    nsteps = 20

    log_dir = 'contour_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with ase.db.connect('structures.db') as db:
        for row in db.select(type='random'):
            row_atoms = row.toatoms()
            initial_energy = relax_structure(row_atoms, steps=20, fmax=0.1)
            delta_e = 1 * row.natoms / nsteps

            dyn = ContourExploration(
                    row_atoms,
                    maxstep = maxstep,
                    angle_limit = angle_limit,
                    logfile = log_dir + '/contour_row_{}.log'.format(row.id)
                    )

            for i in range(nsteps):
                dyn.energy_target += delta_e
                dyn.run(1)

                new_hash = amp.utilities.get_hash(row_atoms)
                db.write(row_atoms, hash=new_hash, type='random_contour')

def relax_structure(xtl, calc=KIM(openkim_pot), optimizer=BFGS, steps=5, fmax=1,
        **kwargs):
    xtl.set_calculator(calc)
    dyn = optimizer(xtl, **kwargs)

    dyn.run(fmax=fmax, steps=steps)

    return xtl.get_potential_energy()

if __name__ == '__main__':
    main()
