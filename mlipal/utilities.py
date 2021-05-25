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
