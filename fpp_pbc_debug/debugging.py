import ase
import amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.utilities import get_hash
import numpy as np


# Set up two unit cells: xtl_centered has the two atoms at the center of the
# cell, and xtl_edges has atoms at opposite ends of the cell. Both should be
# physically identical with periodic boundary conditions.
cell = [6, 4, 4]

xtl_centered = ase.Atoms('Si2',
        positions = [[2,0,0], [4,0,0]],
        cell = cell,
        pbc = [True, True, True])

xtl_edges =  ase.Atoms('Si2',
        positions = [[5,0,0], [1,0,0]],
        cell = cell,
        pbc = [True, True, True])

# Set up descriptor set with Gaussian G2 descriptors.
elements = ['Si']
cutoff = 3
symm_funcs = make_symmetry_functions(elements=elements, type='G2', etas=[1],
        offsets=np.array([0, 0.5, 0.8])*cutoff)
descriptor = Gaussian(Gs=symm_funcs, cutoff=cutoff)

# Compute fingerprints and fingerprintprimes
hash_centered = get_hash(xtl_centered)
hash_edges = get_hash(xtl_edges)
images = {hash_centered: xtl_centered, hash_edges: xtl_edges}
descriptor.calculate_fingerprints(images, calculate_derivatives=True)

fp_centered = descriptor.fingerprints[hash_centered]
fp_edges = descriptor.fingerprints[hash_edges]

fpp_centered = descriptor.fingerprintprimes[hash_centered]
fpp_edges = descriptor.fingerprintprimes[hash_edges]

# Print the results.
print('FINGERPRINTS')
print('Cutoff fully within cell.')
print('PBC:', xtl_centered.pbc)
for fp in fp_centered:
    print(fp)
print()

print('Cutoff wraps around pbc.')
print('PBC:', xtl_edges.pbc)
for fp in fp_edges:
    print(fp)
print()

print('FINGERPRINT PRIMES')
print('Cutoff fully within cell.')
print('PBC:', xtl_centered.pbc)
for key, value in fpp_centered.items():
    print(key, value)
print()

print('Cutoff wraps around pbc.')
print('PBC:', xtl_edges.pbc)
for key, value in fpp_edges.items():
    print(key, value)
