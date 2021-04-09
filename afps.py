import os
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
        PropertyNotImplementedError)
from amp.utilities import get_hash

class AmpFingerPrintSeeker(Calculator):
    '''An ASE calculator which computes a fictional energy and force as a
    function of the difference between one atom's AMP fingerprint and a
    target fingerprint.'''

    implemented_properties = ['energy', 'energies', 'forces',
                              'stress', 'magmom', 'magmoms']

    def __init__(self, amp, force_scale = 1.0, model_calc = None,
            fingerprint_target = None, opt_atom = 0, **kwargs):
        ''' Initialization variables:
        amp: AMP calculator. Needs to contain the descriptors which will be used
        to compute the fingerprints of the Atoms objects.
        force_scale: float. Multiplier for the computed fictional forces and
            energies. Useful for scaling the fictional force to be in line with
            the real forces from model_calc.
        model_calc: ASE calculator. Optional model calculator to mix in real
            forces. This can be used to prevent extremely unfavorable geometries
            (i.e. atomic overlap)
        fingerprint_target: array-like. Target fingerprint to optimize towards.
        opt_atom: integer. Index of the atom whose fingerprint should be
            optimized.

        NOTE: may want to add an optional argument 'atoms', an ASE Atoms object.
        This would allow for a generic AMP calculator to be passed to AFPS
        without an Atoms object attached.
        '''

        Calculator.__init__(self, **kwargs)
        self.amp = amp

        self.model_calc = model_calc
        self.force_scale = force_scale
        self.fingerprint_target = fingerprint_target
        self.opt_atom = opt_atom

    def calculate_fingerprints(self, atoms):
        ''' Calculates fingerprints of atoms.
        Inputs:
            atoms: ASE Atoms object.
        Outputs:
            fingerprints: array-like. Fingerprints of atoms, computed using the
                descriptors from the AMP calculator self.amp. '''

        # XXX Currently calculating the fingerprints (or possible just
        # fingerprintprimes) does not seem to respect periodic boundary
        # conditions. Need to fix this somehow.

        # Creates dictionary that AMP can use to calculate fingerprints
        current_hash = get_hash(atoms)
        hash_dict = {current_hash:atoms}
        self.amp.descriptor.calculate_fingerprints(hash_dict, 
            parallel = self.amp._parallel, log=self.amp._log,
            calculate_derivatives=False)
        fingerprints = self.amp.descriptor.fingerprints[current_hash]
        return fingerprints
        
    def calculate_fingerprintprimes(self,atoms):
        ''' Calculates fingerprint derivatives of atoms.
        Inputs:
            atoms: ASE Atoms object.
        Outputs:
            fingerprintprimes: array-like. Fingerprints of atoms, computed using
                the descriptors from the AMP calculator self.amp. '''

        # XXX Currently calculating the fingerprintprimes does not seem to
        # respect periodic boundary conditions. Need to fix this somehow.

        # Creates dictionary that AMP can use to calculate fingerprints
        current_hash = get_hash(atoms)
        hash_dict = {current_hash:atoms}
        self.amp.descriptor.calculate_fingerprints(hash_dict, 
            parallel =  self.amp._parallel, log=self.amp._log,
            calculate_derivatives=True)
        fingerprintprimes = self.amp.descriptor.fingerprintprimes[current_hash]
        return fingerprintprimes

    def generate_random_fingerprint_targets(self):
        ''' Randomly generates multiple fingerprint targets. '''
        current_fp = self.calculate_fingerprints(self.atoms)
        target_fp = []
        for atom_index in range(len(current_fp)):
            nfp = len(current_fp[atom_index][1])
            target_fp.append((self.atoms[atom_index].symbol,
                list(np.random.rand(nfp)) )  )
        return target_fp

    def initialize(self, atoms):
        ''' Initializes calculator, setting lengths of energies, forces, and
        stress arrays. '''

        #if self.atoms is None:
        #    self.atoms = atoms

        #if self.fingerprint_target is None:
        #    self.fingerprint_target = self.generate_random_fingerprint_targets()
        #else:
        #    self.fingerprint_target = fingerprint_target 

        #self.atoms.set_calculator(self)

        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((3, 3))
        #self.sigma1 = np.empty(len(atoms))

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):

        ''' Calculates fictitious energy and forces based on the difference
        between the fingerprint of the current atoms and the target fingerprint.
        '''

        Calculator.calculate(self, atoms, properties, system_changes)
        #if 'numbers' in system_changes:

        # Initialize calculator
        self.initialize(self.atoms)
        self.energy = 0.0
        self.energies[:] = 0
        #self.sigma1[:] = 0.0
        self.forces[:] = 0.0
        self.stress[:] = 0.0

        # DEBUG: shows that atoms has AFPS calculator attached, self.atoms
        # has no calculator.
        #print('afps atoms is', atoms)
        #print('afps self.atoms is', self.atoms)

        # Real forces.
        # if model_calc is set, will perform a "real" calculation and add the
        # energy and forces to the AFPS calculator's energy and forces.
        if self.model_calc is not None:
            model_atoms = atoms.copy()
            model_atoms.set_calculator(self.model_calc)
            self.energy += model_atoms.get_potential_energy()
            self.forces += model_atoms.get_forces()

        # Ficticious forces.
        # Calculate fingerprints and fingerprint derivatives of current atoms.
        current_fp = self.calculate_fingerprints(self.atoms)
        current_fpp = self.calculate_fingerprintprimes(self.atoms)

        fake_energies = np.zeros(len(atoms))

        # Difference between fingerprint of current atom (to be optimized) and
        # the target fingerprint
        fp_delta = np.array(current_fp[self.opt_atom][1]) \
                - np.array(self.fingerprint_target[1])

        # Defines fictitious energy as the squared magnitude of fp_delta, with
        # some prefactors. force_scale determines the relative magnitude of
        # this fictitious force.
        fake_energy = 0.5 * self.force_scale * np.dot(fp_delta, fp_delta)
        fake_energies[self.opt_atom] = fake_energy
        
        # Add fictitious energies to total energies
        self.energies += fake_energies
        self.energy += np.sum(fake_energies)

        # Compute fictitious forces on each atom
        fake_forces = np.zeros((len(current_fp),3))
        for key in current_fpp.keys():
            # This for loop iterates through pairs of atoms in the structure.
            #
            # Only the force on opt_atom is calculated (key[0]==self.opt_atom),
            # and only with respect to neighboring atoms (key[0] != key[2]),
            # i.e.  dF_A / dx_A is ignored. This is to prevent drifting.
            if (key[0] == self.opt_atom) and (key[0] != key[2]):
                index_B = key[2] # index of surrounding atom

                direction_index = key[4] # direction of the force
                
                partial_f = current_fpp[key] # fingerprint derivative
                
                # Add forces to surrounding atoms
                fake_forces[index_B, direction_index]+=\
                        1.0*self.force_scale*np.dot(fp_delta, partial_f)

        #print('Drift force:')
        #print(np.sum(fake_forces,axis = 0))
        #print()

        # Add fictitious forces to total force
        self.forces += fake_forces

        self.results['energy'] = self.energy
        self.results['energies'] = self.energies
        self.results['free_energy'] = self.energy
        self.results['forces'] = self.forces
