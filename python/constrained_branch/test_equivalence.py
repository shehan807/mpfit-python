"""Verify scalar and vectorized kaisq/dkaisq produce identical results."""

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit._legacy_constrained import (
    kaisq, kaisq_vectorized,
    dkaisq, dkaisq_vectorized,
    setup_from_gdma_record, count_parameters,
    generate_atom_type_labels_from_symmetry,
    generate_quaternary_ammonium_atom_types,
)


def make_gdma_record(smiles, minimize=True):
    """Generate a GDMA record for a molecule."""
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    gdma_settings = GDMASettings()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=minimize
    )
    gdma_record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings
    )
    return molecule, gdma_record


def test_equivalence(name, state, p0):
    """Test scalar vs vectorized for a given state and parameter vector."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Atoms: {len(state.atomtype)}, Params: {len(p0)}")

    obj_s = kaisq(p0, state)
    obj_v = kaisq_vectorized(p0, state)
    obj_match = np.isclose(obj_s, obj_v, rtol=1e-12)
    print(f"  kaisq scalar:     {obj_s:.15e}")
    print(f"  kaisq vectorized: {obj_v:.15e}")
    print(f"  kaisq match: {obj_match}")

    grad_s = dkaisq(p0, state)
    grad_v = dkaisq_vectorized(p0, state)
    max_diff = np.max(np.abs(grad_s - grad_v))
    grad_match = np.allclose(grad_s, grad_v, rtol=1e-12)
    print(f"  dkaisq max diff:  {max_diff:.2e}")
    print(f"  dkaisq allclose:  {grad_match}")

    passed = obj_match and grad_match
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ---- Test 1: Ethanol (small, neutral) ----
molecule, gdma_record = make_gdma_record("CCO")
labels = generate_atom_type_labels_from_symmetry(molecule)
state = setup_from_gdma_record(gdma_record, labels)
n_params = count_parameters(state)
np.random.seed(42)
p0 = np.random.randn(n_params) * 0.01
r1 = test_equivalence("Ethanol (small p0)", state, p0)

# ---- Test 2: Ethanol with large parameters ----
p0_large = np.random.randn(n_params) * 1.0
r2 = test_equivalence("Ethanol (large p0)", state, p0_large)

# ---- Test 3: Ethanol at zero ----
p0_zero = np.zeros(n_params)
r3 = test_equivalence("Ethanol (p0 = 0)", state, p0_zero)

# ---- Test 4: Charged molecule (EMIM, 1+) ----
molecule, gdma_record = make_gdma_record("CCn1cc[n+](C)c1")
labels = generate_atom_type_labels_from_symmetry(molecule)
state = setup_from_gdma_record(gdma_record, labels)
state.molecule_charge = 1.0
n_params = count_parameters(state)
p0 = np.random.randn(n_params) * 0.01
r4 = test_equivalence("EMIM (charged, small p0)", state, p0)

# ---- Test 5: Larger molecule (N4444, 53 atoms) ----
molecule, gdma_record = make_gdma_record("CCCC[N+](CCCC)(CCCC)CCCC", minimize=False)
labels = generate_quaternary_ammonium_atom_types(molecule)
state = setup_from_gdma_record(gdma_record, labels)
state.molecule_charge = 1.0
n_params = count_parameters(state)
p0 = np.random.randn(n_params) * 0.01
r5 = test_equivalence("N4444 (53 atoms, small p0)", state, p0)

# ---- Summary ----
results = [r1, r2, r3, r4, r5]
names = ["Ethanol small", "Ethanol large", "Ethanol zero", "EMIM charged", "N4444 large"]
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for name, passed in zip(names, results):
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
print(f"\n  All passed: {all(results)}")
