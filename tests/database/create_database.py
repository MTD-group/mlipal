import amp.utilities
import ase.db

def create_database(images, filename='ase.db'):
    with ase.db.connect(filename) as db:
        for atoms in images:
            atom_hash = amp.utilities.get_hash(atoms)
            db.write(atoms, hash=atom_hash)
    return db
