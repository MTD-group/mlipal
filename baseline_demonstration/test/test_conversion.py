import ase.db
from amp.descriptor.gaussian import Gaussian
import mlipal

db = ase.db.connect('structures.db')

descriptor = Gaussian(Gs=mlipal.descriptors.two_body_gaussians(['Si'])\
        + mlipal.descriptors.three_body_gaussians(['Si']), cutoff=3)

df = mlipal.utilities.convert_db_to_df(db.select('id<20'), descriptor)

print(df)
