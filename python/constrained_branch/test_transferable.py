"""Test transferable constrained MPFIT across N4444 and P4444.

Demonstrates user-defined atom types for multi-molecule transferability.
Shared carbon/hydrogen types get identical charges across both molecules.
"""
import sys
sys.path.insert(0, '/Users/shehanparmar/Desktop/dev/work/MPFIT_Project/constrained')

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units.elements import SYMBOLS
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit._legacy_constrained import (
    fit_constrained_mpfit,
)


def assign_quaternary_labels(molecule, center_label: str) -> list[str]:
    """Assign transferable atom type labels for a quaternary cation.

    Labels carbons by distance from the central atom (alpha, beta, gamma, delta)
    and hydrogens by their parent carbon type.  The central atom gets a unique
    label; all other types are shared across N4444 / P4444.
    """
    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    # Find the charged center (N+ or P+)
    central = next(
        i for i in range(n_atoms)
        if molecule.atoms[i].atomic_number in (7, 15)
        and int(molecule.atoms[i].formal_charge.magnitude) == 1
    )
    labels[central] = center_label

    # BFS to find carbon distances from center
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

    # Label carbons
    dist_names = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
    for c, d in carbon_dist.items():
        h_count = sum(
            1 for bond in molecule.atoms[c].bonds
            if molecule.atoms[
                bond.atom1_index if bond.atom2_index == c else bond.atom2_index
            ].atomic_number == 1
        )
        labels[c] = "C_CH3" if h_count == 3 else f"C_{dist_names.get(d, f'd{d}')}"

    # Label hydrogens by parent
    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            parent = next(
                bond.atom1_index if bond.atom2_index == i else bond.atom2_index
                for bond in molecule.atoms[i].bonds
            )
            if labels[parent]:
                labels[i] = f"H_{labels[parent].replace('C_', '')}"

    return labels


# ---- Build GDMA records ----
gdma_settings = GDMASettings()

print("=" * 70)
print("Transferable MPFIT: N4444 + P4444")
print("=" * 70)

records = []
all_labels = []
molecules = []

for name, smiles, center_label in [
    ("N4444", "CCCC[N+](CCCC)(CCCC)CCCC", "N_center"),
    ("P4444", "CCCC[P+](CCCC)(CCCC)CCCC", "P_center"),
]:
    print(f"\n--- {name} ---")
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(mol)
    conformer, multipoles = Psi4GDMAGenerator.generate(
        mol, conformer, gdma_settings, minimize=False,
    )
    record = MoleculeGDMARecord.from_molecule(mol, conformer, multipoles, gdma_settings)
    records.append(record)
    molecules.append(mol)

    labels = assign_quaternary_labels(mol, center_label)
    all_labels.extend(labels)

    print(f"  Atoms: {mol.n_atoms}")
    for i, lbl in enumerate(labels):
        print(f"    {i:2d} ({SYMBOLS[mol.atoms[i].atomic_number]:2s}): {lbl}")

# ---- Show shared types ----
print(f"\n--- Transferable Labels ({len(all_labels)} total) ---")
from collections import Counter
counts = Counter(all_labels)
print("  Shared types (appear in both molecules):")
n4_atoms = molecules[0].n_atoms
for lbl, cnt in sorted(counts.items()):
    in_n4 = lbl in all_labels[:n4_atoms]
    in_p4 = lbl in all_labels[n4_atoms:]
    if in_n4 and in_p4:
        print(f"    {lbl}: {cnt} atoms")

# ---- Fit ----
print("\n--- Fitting ---")
total_charge = 2.0  # both are +1
result = fit_constrained_mpfit(records, all_labels, molecule_charge=total_charge)

# ---- Results ----
print(f"\nOptimization success: {result['success']}")
print(f"Objective: {result['objective']:.6e}")

# Split charges by molecule
n4_charges = result["qstore"][:n4_atoms]
p4_charges = result["qstore"][n4_atoms:]
n4_labels = all_labels[:n4_atoms]
p4_labels = all_labels[n4_atoms:]

for name, charges, labels in [("N4444", n4_charges, n4_labels), ("P4444", p4_charges, p4_labels)]:
    print(f"\n--- {name} Charges ---")
    for i, (lbl, q) in enumerate(zip(labels, charges)):
        print(f"  {i:2d} ({lbl:>10s}): {q:+.6f}")
    print(f"  Sum: {np.sum(charges):+.6f}")

# Verify shared types got identical charges
print("\n--- Transferability Check ---")
unique_charges = {}
for lbl, q in zip(all_labels, result["qstore"]):
    unique_charges.setdefault(lbl, []).append(q)

all_ok = True
for lbl, qs in sorted(unique_charges.items()):
    if len(qs) > 1:
        max_diff = max(qs) - min(qs)
        ok = max_diff < 1e-10
        if not ok:
            all_ok = False
        print(f"  {lbl:>10s}: q={qs[0]:+.6f}  (n={len(qs)}, max_diff={max_diff:.2e}) {'PASS' if ok else 'FAIL'}")

print(f"\nAll transferable: {all_ok}")
