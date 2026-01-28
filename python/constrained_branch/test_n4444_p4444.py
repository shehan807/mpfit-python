import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit._legacy_constrained import (
    fit_transferable_charges, generate_quaternary_ammonium_atom_types
)
import time

print('=== N4444 + P4444 Transferable Charge Fitting ===\n')

# Use HF/6-31G* for faster calculations
gdma_settings = GDMASettings(method='hf', basis='6-31G*')
print(f'Theory level: {gdma_settings.method}/{gdma_settings.basis}\n')

# N4444
print('Generating GDMA for N4444...')
t0 = time.time()
n4444 = Molecule.from_smiles('CCCC[N+](CCCC)(CCCC)CCCC')
n4444.generate_conformers(n_conformers=1)
[conf_n] = extract_conformers(n4444)
conf_n, mult_n = Psi4GDMAGenerator.generate(n4444, conf_n, gdma_settings, minimize=False, n_threads=8)
gdma_n = MoleculeGDMARecord.from_molecule(n4444, conf_n, mult_n, gdma_settings)
print(f'  Done in {time.time()-t0:.1f}s')

# P4444
print('Generating GDMA for P4444...')
t0 = time.time()
p4444 = Molecule.from_smiles('CCCC[P+](CCCC)(CCCC)CCCC')
p4444.generate_conformers(n_conformers=1)
[conf_p] = extract_conformers(p4444)
conf_p, mult_p = Psi4GDMAGenerator.generate(p4444, conf_p, gdma_settings, minimize=False, n_threads=8)
gdma_p = MoleculeGDMARecord.from_molecule(p4444, conf_p, mult_p, gdma_settings)
print(f'  Done in {time.time()-t0:.1f}s')

# Fit transferable charges
n_labels = generate_quaternary_ammonium_atom_types(n4444)
p_labels = generate_quaternary_ammonium_atom_types(p4444)

print(f'\nN4444: {n4444.n_atoms} atoms, P4444: {p4444.n_atoms} atoms')
print(f'Unique atom types: {len(set(n_labels + p_labels))}\n')

result = fit_transferable_charges(
    gdma_records=[gdma_n, gdma_p],
    atom_type_labels=n_labels + p_labels,
    molecule_charge=2.0,
    verbose=True,
)

print('\n=== Unique Charges ===')
for label, q in sorted(result['unique_charges'].items()):
    print(f'  {label:10s}: {q:+.6f}')

print(f"\nN4444 sum: {np.sum(result['charges_by_molecule'][0]):+.4f}")
print(f"P4444 sum: {np.sum(result['charges_by_molecule'][1]):+.4f}")
