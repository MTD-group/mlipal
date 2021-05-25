def convert_db_to_df(db, descriptor=None, columns=None):
    ''' Converts an ASE database into a dataframe. If an AMP descriptor is
    given, the dataframe will contain the fingerprints as well.

    db: ASE Database.
        Database to be converted into a dataframe.

    descriptor: AMP Descriptor.
        If given, fingerprints will be computed for each Atoms object in the
        database.

    columns: list.
        A list of column names to be included in the dataframe. If not given,
        the dataframe will include columns for energy, forces, and all extra
        columns in the db.'''

    count = 0
    for row in db.select():
        descriptor.calculate_fingerprints({row.hash: row.toatoms()})

        hashes = np.asarray(row.hash).reshape(1,)

        structure_type = np.asarray(row.type).reshape(1,)

        energy = row.energy / row.natoms
        energy = np.array(energy).reshape(1,)

        forces = row.forces

        for i in range(row.natoms):
            atoms_fp = descriptor.fingerprints[row.hash][i]
            element = np.asarray(atoms_fp[0]).reshape(1,)
            feature = np.array(atoms_fp[1])
            force = forces[i]
            atom_info = np.concatenate([hashes, structure_type, energy, force,
                element, feature])
            atom_info = np.reshape(atom_info, (1, atom_info.shape[0]))
            if count==0:
                atom_data = atom_info
                count = 1
            else:
                atom_data = np.append(atom_data, atom_info, axis=0)     

    df = pd.DataFrame(atom_data)  
    df = df.rename(columns={0: "hash", 1:"type", 2: "element", 3: "energy", 4: "force0", 5: "force1", 6: "force2"})             

    return df