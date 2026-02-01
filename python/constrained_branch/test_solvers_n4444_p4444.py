"""Compare SciPySolver and JAXSolver on N4444 + P4444.

Reports JIT compilation time separately from optimization time.
"""

import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord

from openff_pympfit.mpfit._legacy_constrained_jax_pure import (
    setup_from_gdma_records,
    SciPySolver,
    JAXSolver,
    count_parameters,
)


def assign_quaternary_labels(molecule, center_label):
    """Assign transferable atom type labels for a quaternary cation."""
    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    central = next(
        i for i in range(n_atoms)
        if molecule.atoms[i].atomic_number in (7, 15)
        and int(molecule.atoms[i].formal_charge.magnitude) == 1
    )
    labels[central] = center_label

    visited = {central}
    queue = [(central, 0)]
    carbon_dist = {}

    while queue:
        current, dist = queue.pop(0)
        for bond in molecule.atoms[current].bonds:
            nbr = bond.atom1_index if bond.atom2_index == current else bond.atom2_index
            if nbr in visited:
                continue
            visited.add(nbr)
            if molecule.atoms[nbr].atomic_number == 6:
                carbon_dist[nbr] = dist + 1
                queue.append((nbr, dist + 1))

    dist_names = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
    for c, d in carbon_dist.items():
        h_count = sum(
            1 for bond in molecule.atoms[c].bonds
            if molecule.atoms[
                bond.atom1_index if bond.atom2_index == c else bond.atom2_index
            ].atomic_number == 1
        )
        labels[c] = "C_CH3" if h_count == 3 else f"C_{dist_names.get(d, f'd{d}')}"

    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            parent = next(
                bond.atom1_index if bond.atom2_index == i else bond.atom2_index
                for bond in molecule.atoms[i].bonds
            )
            if labels[parent]:
                labels[i] = f"H_{labels[parent].replace('C_', '')}"

    return labels


def make_records():
    """Generate GDMA records for N4444 and P4444."""
    gdma_settings = GDMASettings()
    records = []
    all_labels = []

    for name, smiles, center_label in [
        ("N4444", "CCCC[N+](CCCC)(CCCC)CCCC", "N_center"),
        ("P4444", "CCCC[P+](CCCC)(CCCC)CCCC", "P_center"),
    ]:
        print(f"  Generating {name}...")
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        mol.generate_conformers(n_conformers=1)
        [conformer] = extract_conformers(mol)
        conformer, multipoles = Psi4GDMAGenerator.generate(
            mol, conformer, gdma_settings, minimize=False,
        )
        record = MoleculeGDMARecord.from_molecule(mol, conformer, multipoles, gdma_settings)
        records.append(record)
        labels = assign_quaternary_labels(mol, center_label)
        all_labels.extend(labels)
        print(f"    {mol.n_atoms} atoms")

    return records, all_labels


def main():
    print("=" * 70)
    print("  Solver Comparison — N4444 + P4444")
    print("=" * 70)

    print("\n  Generating GDMA records (Psi4)...")
    records, all_labels = make_records()

    state = setup_from_gdma_records(records, all_labels)
    state.molecule_charge = 2.0
    n_params = count_parameters(state)
    n_atoms = len(all_labels)
    print(f"\n  Atoms: {n_atoms}, Parameters: {n_params}")

    # --- SciPySolver ---
    print("\n--- SciPySolver (numpy) ---")
    t0 = time.perf_counter()
    solver_np = SciPySolver(state)
    t_setup_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_np = solver_np.optimize()
    t_opt_np = time.perf_counter() - t0

    print(f"  Setup:        {t_setup_np:.3f}s")
    print(f"  Optimization: {t_opt_np:.3f}s")
    print(f"  Objective:    {res_np['objective']:.12e}")

    # --- JAXSolver ---
    print("\n--- JAXSolver (JIT + auto-diff) ---")
    t0 = time.perf_counter()
    solver_jax = JAXSolver(state)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_jax = solver_jax.optimize()
    t_opt_jax = time.perf_counter() - t0

    print(f"  JIT compile:  {t_compile:.3f}s")
    print(f"  Optimization: {t_opt_jax:.3f}s")
    print(f"  Objective:    {res_jax['objective']:.12e}")

    # --- Comparison ---
    max_charge_diff = np.max(np.abs(res_np["qstore"] - res_jax["qstore"]))
    obj_rel_diff = abs(res_np["objective"] - res_jax["objective"]) / abs(res_np["objective"])

    print(f"\n--- Comparison ---")
    print(f"  Max charge diff:   {max_charge_diff:.2e}")
    print(f"  Obj relative diff: {obj_rel_diff:.2e}")
    print(f"  Speedup (opt only): {t_opt_np / t_opt_jax:.2f}x")

    charges_close = np.allclose(res_np["qstore"], res_jax["qstore"], atol=1e-3)
    obj_close = obj_rel_diff < 1e-4
    passed = charges_close and obj_close
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    # --- Per-atom charge table ---
    print(f"\n{'=' * 70}")
    print(f"  Per-Atom Charges")
    print(f"{'=' * 70}")
    print(f"  {'Atom':>4} {'Label':<12} {'SciPy':>12} {'JAX':>12} {'Diff':>12}")
    print(f"  {'-' * 56}")
    for i in range(n_atoms):
        diff = res_np["qstore"][i] - res_jax["qstore"][i]
        print(f"  {i:>4} {all_labels[i]:<12} {res_np['qstore'][i]:>12.6f} {res_jax['qstore'][i]:>12.6f} {diff:>12.2e}")

    return passed


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
