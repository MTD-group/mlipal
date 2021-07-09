from ase.calculators.kim import KIM
from ase.optimize import BFGS

openkim_pot = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'

def relax_structure(xtl, calc=KIM(openkim_pot), optimizer=BFGS, steps=5, fmax=5,
        **kwargs):
    xtl.set_calculator(calc)
    dyn = optimizer(xtl, **kwargs)

    dyn.run(fmax=fmax, steps=steps)
