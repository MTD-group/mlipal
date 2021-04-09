from afps import AmpFingerPrintSeeker
from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.utilities import Logger, make_filename, hash_images, get_hash
from ase import io
from ase.calculators.kim import KIM
import numpy as np

kim_calc = 'Tersoff_LAMMPS_Tersoff_1988T3_Si__MO_186459956893_003'
#my_model_calc = KIM(kim_calc)
my_model_calc = None

# Generates descriptors
cutoff = 3
elements = ['Si']
g2_etas = [1, 20]
g2_offsets = offsets=np.array([0, 0.3, 0.5, 0.68, 0.8])*cutoff

g2_symm_funcs = make_symmetry_functions(elements=elements, type='G2', etas=g2_etas,
        offsets=g2_offsets)

symm_funcs = g2_symm_funcs
descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

model = NeuralNetwork()

log_file = 'amp_d_calc.log'
MLIP = Amp(descriptor = descriptor, model=model)
MLIP.cores['localhost'] = cores = 1
MLIP._log = Logger(make_filename('',log_file))

###############
#pair = ['Al', 'Al']
#
#def generate_atoms(radius_scale, vacuum_scale = 4):
#    from ase import Atoms
#    from ase.data import atomic_numbers, covalent_radii
#    def generate_radius(radius_scale=1.0):
#        r0 = covalent_radii[atomic_numbers[pair[0]]]
#        r1 = covalent_radii[atomic_numbers[pair[1]]]
#        return radius_scale*(r0+r1)
#
#    #vacuum = vacuum_scale*generate_radius(radius_scale=1.0)
#    
#    vacuum = 8*cut_off_radius
#
#    radius = generate_radius(radius_scale = radius_scale)
#
#    
#    atoms = Atoms(pair, positions=[[0, 0, 0],[radius,0,0]])

#    atoms.center(vacuum=vacuum/2) # only need half since it's periodic
#    atoms.pbc = True
#    return atoms
#
################

#atoms = generate_atoms(0.9)
#target_atoms = generate_atoms(1.5)

# NOTE: This reads the atoms file. 
# Atom 41 is near a boundary, and its distance to atom 57 (which is the one
# being optimized in AFPS) is 2.87A including PBC. However, it is on the
# opposite side of the unit cell. This means that if AMP is taking into account
# PBC when calculating fingerprint primes for atom 57, there should be a
# non-zero fingerprint prime (and thus non-zero force) for atom 41.
# I add jitter to atom 41 to ensure that it is not already at a local minimum.
atoms = io.read('0.POSCAR.vasp')
#jitter = np.zeros((len(atoms), 3))
#jitter[41] = np.random.rand(3) * 0.05
#atoms.set_positions(atoms.get_positions() + jitter)

### Setting target fingerprint
target_atoms = io.read('1.POSCAR.vasp')
target_hash = get_hash(target_atoms)
descriptor.calculate_fingerprints({target_hash: target_atoms})
target_fp = descriptor.fingerprints[target_hash][0]

#io.write('initial.CONTCAR', atoms, format = 'vasp',vasp5=True )
#io.write('target.CONTCAR', target_atoms, format = 'vasp',vasp5=True )

my_afps = AmpFingerPrintSeeker(amp = MLIP, fingerprint_target=target_fp,
        force_scale = 1.0, model_calc = my_model_calc )
atoms.set_calculator(my_afps)

#my_afps.fingerprint_targets = my_afps.calculate_fingerprints(target_atoms)

#print(atoms.get_calculator())

if __name__ == '__main__':

    energies = atoms.get_potential_energies()
    forces = atoms.get_forces(atoms)

    print('Calculator for atoms:')
    print(atoms.get_calculator())

    #print('IP energy and forces')
    #print(my_model_calc.get_potential_energy(atoms.copy()))
    #print(my_model_calc.get_forces(atoms.copy()))

    print('Calculator for atoms:')
    print(atoms.get_calculator())

    print('IP+fake energy, energies, and forces')
    initial_energy = atoms.get_potential_energy()
    #energies = atoms.get_potential_energies()
    #forces = atoms.get_forces(atoms)
    print(initial_energy)
    #print(energies)
    #print(forces[1])

    print('Calculator for atoms:')
    print(atoms.get_calculator())


    ### now we optimize with fake forces
    from ase.optimize import BFGS
    dyn = BFGS(atoms, trajectory='seeker.traj')
    #dyn.max_steps = 100
    dyn.run(fmax = 0.01)

    ####### now print the optimized  data #######
    #print('Opt. IP energy and forces')
    #print(my_model_calc.get_potential_energy(atoms.copy()))
    #print(my_model_calc.get_forces(atoms.copy()))
    #print('Calculator for atoms:')
    #print(atoms.get_calculator())

    print('Opt. IP+fake energy and forces')
    energy = atoms.get_potential_energy()
    energies = atoms.get_potential_energies()
    forces = atoms.get_forces(atoms)
    print(energy)
    print('Force on atom 41:')
    print(forces[41])

    print('Energy difference, opt - init')
    print(energy, initial_energy)
    print(energy - initial_energy)

    print('Opt. drift force')
    print(np.sum(forces, axis=0))

    io.write('output.vasp', atoms, format='vasp', vasp5=True)
