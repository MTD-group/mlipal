import ase.db
import numpy as np

def find_closest_fingerprint(target_fp, descriptor,
        structuredb='structures.db'):
    ''' Searches structure database to find the structure and atom index with
    the closest fingerprint to the target_fp. '''

    # Euclidean difference between target fingerprint and best match
    match_delta = -1

    db = ase.db.connect(structuredb)
    for row in db.select():
        for i, fp in enumerate(descriptor.fingerprints[row.hash]):
            delta_fp = np.array(target_fp) - np.array(fp[1])
            delta_norm = np.linalg.norm(delta_fp)
            if delta_norm < match_delta:
                match_delta = delta_norm
                match_hash = row.hash
                match_index = i
            elif match_delta == -1:
                match_delta = delta_norm

    return (match_hash, match_index), match_delta

if __name__ == '__main__':
    from amp.descriptor.gaussian import Gaussian
    import mlipal

    db = ase.db.connect('structures.db')
    a = db.get_atoms(id=1)

    symm_funcs = mlipal.descriptors.two_body_gaussians(a)\
            + mlipal.descriptors.three_body_gaussians(a)
    descriptor = Gaussian(Gs=symm_funcs, cutoff=3)

    for row in db.select():
        descriptor.calculate_fingerprints({row.hash: row.toatoms()})

    target_fp = np.random.rand(37)
    (h, i), d = find_closest_fingerprint(target_fp, descriptor)

    atoms = db.get_atoms(hash=h)

    print(h, atoms)
    print(i)
    print('Target:', target_fp)
    print('Match:', descriptor.fingerprints[h][i])
    print(d)
