from ase.calculators.kim import KIM
import ase.db

def compute_energies(calculator, db_file='structures.db'):
    with ase.db.connect(db_file) as db:
        for row in db.select():
            atoms = row.toatoms()
            atoms.set_calculator(calculator)
            atoms.get_potential_energy()
            atoms.get_forces()
            db.update(row.id, atoms=atoms)

def main():
    kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'
    calc = KIM(kim_calc)

    compute_energies(calc, 'structures_no_random.db')

if __name__ == '__main__':
    main()
