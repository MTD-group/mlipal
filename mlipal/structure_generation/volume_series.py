import ase
from ase.build import bulk
import numpy as np
import os

def volume_series(equilibrium_structure=None, vol_mults = np.linspace(0.6, 2, 50)):
    ''' Generates structures in a volume series.

    equilibrium_structure: Atoms object.
        Equilibrium structure relative to which volumes will be scaled
    vol_mults: array-like.
        List of scaling factors. This scales the cell unit vectors, not the
        volume itself!

    Returns:
    xtls: list. Contains Atoms objects with scaled volumes.'''

    # Si crystal structure is default
    if equilibrium_structure == None:
        equilibrium_structure = bulk('Si', 'diamond', a=5.43)

    xtls = []
    for v in vol_mults:
        # Create copy of equilibrium crystal to scale by volume multiplier
        xtl_scaled = equilibrium_structure.copy()
        xtl_scaled.set_cell(xtl_scaled.get_cell() * v**(1/3), scale_atoms=True)

        xtls.append(xtl_scaled)

    return xtls

def main():
    # This sets up a series of lattice constant multipliers that is biased
    # towards the ground state lattice constant (i.e. v=1)
    v1 = np.linspace(0.6, 0.8, 81)
    v2 = np.linspace(0.8, 1.2, 401)
    v3 = np.linspace(1.2, 2, 201)
    vol_mults = np.concatenate( (v1, v2, v3) )

    xtls = generate_volume_series(vol_mults)

    for i, vol in enumerate(vol_mults):
        xtl = xtls[i]
        xtl.write(os.path.join(data_dir, '{:.2f}.traj'.format(vol)))

if __name__ == '__main__':
    main()
