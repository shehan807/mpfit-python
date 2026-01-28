"""Benchmark JAX JIT vs numpy vectorized optimization."""
# Enable JAX float64 BEFORE any JAX import
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys

from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit._legacy_constrained import (
    setup_from_multiple_gdma_records,
    generate_quaternary_ammonium_atom_types,
    optimize_constrained,
    optimize_constrained_jit,
    count_parameters,
    kaisq_vectorized,
    kaisq_jit,
    _make_jit_functions,
)


def print_charges(labels, qstore, title):
    """Print per-atom charges and unique charge summary."""
    print(f"\n--- {title}: Per-Atom Charges ---")
    for i, (label, q) in enumerate(zip(labels, qstore)):
        print(f"  {i:3d} ({label:12s}): {q:+.6f}")

    print(f"\n--- {title}: Unique Charges ---")
    seen = {}
    for label, q in zip(labels, qstore):
        if label not in seen:
            seen[label] = q
    for label, q in sorted(seen.items()):
        print(f"  {label:12s}: {q:+.6f}")

    print(f"  Total charge: {np.sum(qstore):+.6f}")


def test_n4444_p4444_jit():
    """Benchmark JIT vs vectorized on N4444 + P4444."""
    print("=" * 60)
    print("N4444 + P4444: JIT benchmark")
    print("=" * 60)

    gdma_settings = GDMASettings(method='hf', basis='6-31G*')

    # N4444
    print("Generating GDMA for N4444...")
    t0 = time.time()
    n4444 = Molecule.from_smiles('CCCC[N+](CCCC)(CCCC)CCCC')
    n4444.generate_conformers(n_conformers=1)
    [conf_n] = extract_conformers(n4444)
    conf_n, mult_n = Psi4GDMAGenerator.generate(n4444, conf_n, gdma_settings, minimize=False, n_threads=8)
    gdma_n = MoleculeGDMARecord.from_molecule(n4444, conf_n, mult_n, gdma_settings)
    print(f"  Done in {time.time()-t0:.1f}s")

    # P4444
    print("Generating GDMA for P4444...")
    t0 = time.time()
    p4444 = Molecule.from_smiles('CCCC[P+](CCCC)(CCCC)CCCC')
    p4444.generate_conformers(n_conformers=1)
    [conf_p] = extract_conformers(p4444)
    conf_p, mult_p = Psi4GDMAGenerator.generate(p4444, conf_p, gdma_settings, minimize=False, n_threads=8)
    gdma_p = MoleculeGDMARecord.from_molecule(p4444, conf_p, mult_p, gdma_settings)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Setup
    n_labels = generate_quaternary_ammonium_atom_types(n4444)
    p_labels = generate_quaternary_ammonium_atom_types(p4444)
    all_labels = n_labels + p_labels

    state_vec = setup_from_multiple_gdma_records([gdma_n, gdma_p], all_labels)
    state_vec.molecule_charge = 2.0

    state_jit = setup_from_multiple_gdma_records([gdma_n, gdma_p], all_labels)
    state_jit.molecule_charge = 2.0

    n_params = count_parameters(state_vec)
    n_n = n4444.n_atoms
    n_p = p4444.n_atoms
    print(f"\nTotal atoms: {n_n + n_p} (N4444: {n_n}, P4444: {n_p})")
    print(f"Free parameters: {n_params}")
    print(f"Sites: {state_vec.xyzmult.shape[0]}")

    # Verify initial objectives match
    p0 = np.zeros(n_params)
    obj_vec_init = kaisq_vectorized(p0, state_vec)
    jit_fns = _make_jit_functions(state_jit)
    # Reset state_jit since kaisq_vectorized modified state_vec
    state_jit2 = setup_from_multiple_gdma_records([gdma_n, gdma_p], all_labels)
    state_jit2.molecule_charge = 2.0
    obj_jit_init = kaisq_jit(p0, state_jit2, jit_fns)
    print(f"\nInitial obj (vec): {obj_vec_init:.10e}")
    print(f"Initial obj (JIT): {obj_jit_init:.10e}")
    print(f"Match: {np.isclose(obj_vec_init, obj_jit_init, rtol=1e-12)}")

    # Re-setup clean states
    state_vec = setup_from_multiple_gdma_records([gdma_n, gdma_p], all_labels)
    state_vec.molecule_charge = 2.0
    state_jit = setup_from_multiple_gdma_records([gdma_n, gdma_p], all_labels)
    state_jit.molecule_charge = 2.0

    # Vectorized
    print("\n--- Vectorized optimization ---")
    result_vec = optimize_constrained(state_vec, verbose=True)

    # JIT
    print("\n--- JIT optimization ---")
    result_jit = optimize_constrained_jit(state_jit, verbose=True)

    # Print charges
    print_charges(n_labels, result_vec['qstore'][:n_n], "Vectorized N4444")
    print_charges(p_labels, result_vec['qstore'][n_n:], "Vectorized P4444")
    print_charges(n_labels, result_jit['qstore'][:n_n], "JIT N4444")
    print_charges(p_labels, result_jit['qstore'][n_n:], "JIT P4444")

    # Comparison
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print(f"Final obj (vec): {result_vec['objective']:.10e}")
    print(f"Final obj (JIT): {result_jit['objective']:.10e}")
    print(f"Max charge diff: {np.max(np.abs(result_vec['qstore'] - result_jit['qstore'])):.2e}")

    # Side-by-side unique charges
    seen_vec, seen_jit = {}, {}
    for label, q in zip(all_labels, result_vec['qstore']):
        if label not in seen_vec:
            seen_vec[label] = q
    for label, q in zip(all_labels, result_jit['qstore']):
        if label not in seen_jit:
            seen_jit[label] = q

    print(f"\n{'Label':12s}  {'Vectorized':>12s}  {'JIT':>12s}  {'Diff':>12s}")
    print("-" * 52)
    for label in sorted(seen_vec.keys()):
        qv = seen_vec[label]
        qj = seen_jit[label]
        print(f"{label:12s}  {qv:+12.6f}  {qj:+12.6f}  {abs(qv-qj):12.2e}")


if __name__ == "__main__":
    test_n4444_p4444_jit()
