"""Constrained charge fitting for imidazolium cation family.

MMIM  = 1,3-dimethylimidazolium       [+1]
EMIM  = 1-ethyl-3-methylimidazolium   [+1]
BMIM  = 1-butyl-3-methylimidazolium   [+1]
C6MIM = 1-hexyl-3-methylimidazolium   [+1]

Transferable atom types are assigned based on the imidazolium ring topology:
  Ring:  N_me (N with methyl), N_alk (N with longer chain), C2, C4, C5
  Chain: C_me (methyl), C_alpha, C_beta, C_gamma, C_delta, C_eps, C_term
  H's:   H_<parent_label>

For MMIM (symmetric), both N's are N_me and both ring backbone C's are C4.
"""

import time

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units.elements import SYMBOLS
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord

from openff_pympfit.mpfit._legacy_constrained_jax_sumchg import (
    setup_from_gdma_records,
    optimize_constrained,
    count_parameters,
)


# --- SMILES ---
MOLECULES = {
    "MMIM":  "C[n+]1ccn(C)c1",
    "EMIM":  "CC[n+]1ccn(C)c1",
    "BMIM":  "CCCC[n+]1ccn(C)c1",
    "C6MIM": "CCCCCC[n+]1ccn(C)c1",
}


def assign_imidazolium_labels(molecule, name):
    """Assign transferable atom type labels for an imidazolium cation.

    Identifies the 5-membered ring by connectivity (OpenFF doesn't mark
    aromaticity), then assigns labels based on topology.
    """
    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    # Build adjacency for heavy atoms
    adj = {i: set() for i in range(n_atoms)}
    for i in range(n_atoms):
        for bond in molecule.atoms[i].bonds:
            nbr = bond.atom1_index if bond.atom2_index == i else bond.atom2_index
            adj[i].add(nbr)

    # 1. Find the two ring nitrogens
    ring_nitrogens = [
        i for i in range(n_atoms) if molecule.atoms[i].atomic_number == 7
    ]
    if len(ring_nitrogens) != 2:
        raise ValueError(f"{name}: expected 2 ring nitrogens, got {len(ring_nitrogens)}")

    # 2. Find the 5-membered ring: both N's share it.
    #    Ring carbons are the heavy-atom neighbors of N that are also in the ring.
    #    The ring is: N1 - C - C - N2 - C (back to N1)
    n1, n2 = ring_nitrogens

    # Heavy-atom neighbors of each N (exclude H)
    def heavy_nbrs(idx):
        return {n for n in adj[idx] if molecule.atoms[n].atomic_number != 1}

    # Find ring carbons: C bonded to both N's is C2
    # C bonded to one N and to the other ring C is C4 or C5
    n1_c_nbrs = {n for n in heavy_nbrs(n1) if molecule.atoms[n].atomic_number == 6}
    n2_c_nbrs = {n for n in heavy_nbrs(n2) if molecule.atoms[n].atomic_number == 6}

    # C2 is the carbon bonded to BOTH nitrogens
    c2_candidates = n1_c_nbrs & n2_c_nbrs
    if len(c2_candidates) != 1:
        raise ValueError(f"{name}: expected 1 C2 (bonded to both N), got {c2_candidates}")
    c2_idx = c2_candidates.pop()

    # The other ring carbons: each N has one ring-C neighbor besides C2
    # These two C's are bonded to each other (C4-C5 bond in the ring)
    ring_c_from_n1 = n1_c_nbrs - {c2_idx}
    ring_c_from_n2 = n2_c_nbrs - {c2_idx}

    # Each N may also bond to non-ring carbons (substituent).
    # Ring C's are the ones that bond to the OTHER ring C.
    # Find by checking: which of n1's C-neighbors (besides C2) bonds to one of n2's C-neighbors?
    c4_idx = c5_idx = None
    for ca in ring_c_from_n1:
        for cb in ring_c_from_n2:
            if cb in adj[ca]:
                # ca is bonded to n1 and to cb; cb is bonded to n2
                c5_idx = ca  # bonded to n1
                c4_idx = cb  # bonded to n2
                break
        if c4_idx is not None:
            break

    if c4_idx is None or c5_idx is None:
        raise ValueError(f"{name}: could not identify ring C4/C5")

    ring_atoms = {n1, n2, c2_idx, c4_idx, c5_idx}

    # 3. For each N, find the substituent chain (non-ring carbons)
    def get_chain(n_idx):
        chain = []
        # Find non-ring carbon neighbor
        for nbr in heavy_nbrs(n_idx):
            if molecule.atoms[nbr].atomic_number == 6 and nbr not in ring_atoms:
                chain.append(nbr)
                break
        if not chain:
            return chain
        visited = ring_atoms | {chain[0]}
        current = chain[0]
        while True:
            found = False
            for nbr in adj[current]:
                if nbr not in visited and molecule.atoms[nbr].atomic_number == 6:
                    chain.append(nbr)
                    visited.add(nbr)
                    current = nbr
                    found = True
                    break
            if not found:
                break
        return chain

    chain_n1 = get_chain(n1)
    chain_n2 = get_chain(n2)

    # 4. Shorter chain → N_me; longer → N_alk. Symmetric if equal.
    is_symmetric = len(chain_n1) == len(chain_n2)
    if len(chain_n1) <= len(chain_n2):
        n_me_idx, n_alk_idx = n1, n2
        me_chain, alk_chain = chain_n1, chain_n2
        # c5 bonded to n_me, c4 bonded to n_alk
        # but we assigned c5=bonded to n1, c4=bonded to n2 above
        # so if n_me=n1, c5 is correct
    else:
        n_me_idx, n_alk_idx = n2, n1
        me_chain, alk_chain = chain_n2, chain_n1
        # Swap C4/C5: c4 should be bonded to n_alk, c5 to n_me
        c4_idx, c5_idx = c5_idx, c4_idx

    # 5. Label ring atoms
    labels[n_me_idx] = "N_me"
    labels[n_alk_idx] = "N_me" if is_symmetric else "N_alk"
    labels[c2_idx] = "C2"
    labels[c4_idx] = "C4" if not is_symmetric else "C4"
    labels[c5_idx] = "C5" if not is_symmetric else "C4"

    # 6. Methyl on N_me
    if me_chain:
        labels[me_chain[0]] = "C_me"

    # 7. Alkyl chain on N_alk
    chain_names = ["C_alpha", "C_beta", "C_gamma", "C_delta", "C_eps", "C_zeta"]
    if is_symmetric:
        for c_idx in alk_chain:
            labels[c_idx] = "C_me"
    else:
        for j, c_idx in enumerate(alk_chain):
            if j == len(alk_chain) - 1:
                labels[c_idx] = "C_term"
            elif j < len(chain_names):
                labels[c_idx] = chain_names[j]
            else:
                labels[c_idx] = f"C_ch{j}"

    # 8. Hydrogen labels from parent
    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            for bond in molecule.atoms[i].bonds:
                parent = bond.atom1_index if bond.atom2_index == i else bond.atom2_index
                if labels[parent]:
                    labels[i] = "H_" + labels[parent]
                    break

    # Verify
    for i in range(n_atoms):
        if not labels[i]:
            raise ValueError(
                f"{name}: atom {i} ({SYMBOLS[molecule.atoms[i].atomic_number]}) unlabeled"
            )

    return labels


def make_record(smiles):
    """Generate a GDMA record from SMILES."""
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    gdma_settings = GDMASettings()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=True,
    )
    record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings,
    )
    return molecule, record


def main():
    print("=" * 70)
    print("  Imidazolium Cation Family — Constrained Charge Fitting")
    print("  MMIM / EMIM / BMIM / C6MIM   (total charge = +4)")
    print("=" * 70)

    # --- Generate GDMA records ---
    all_data = []
    for name, smiles in MOLECULES.items():
        print(f"\n  Generating {name} ({smiles})...", end="", flush=True)
        t0 = time.perf_counter()
        mol, record = make_record(smiles)
        labels = assign_imidazolium_labels(mol, name)
        dt = time.perf_counter() - t0
        all_data.append((name, mol, record, labels))
        print(f" {mol.n_atoms} atoms  ({dt:.1f}s)")
        print(f"    Labels: {labels}")

    # --- Combine records ---
    records = [d[2] for d in all_data]
    all_labels = []
    for d in all_data:
        all_labels.extend(d[3])

    state = setup_from_gdma_records(records, all_labels)
    state.molecule_charges = [1.0] * len(records)  # each cation is +1
    state.molecule_charge = 4.0  # total (for reference)
    state.conchg = 1.0  # charge conservation penalty weight
    n_params = count_parameters(state)
    n_atoms = len(all_labels)

    print(f"\n{'=' * 70}")
    print(f"  Combined system: {n_atoms} atoms, {n_params} parameters")
    print(f"  Total charge constraint: {state.molecule_charge}")

    # --- Unique atom types ---
    unique = sorted(set(all_labels))
    print(f"  Unique atom types: {len(unique)}")
    for lbl in unique:
        count = all_labels.count(lbl)
        print(f"    {lbl:<16} x{count}")

    # --- Optimize ---
    print(f"\n{'=' * 70}")
    print("  Running optimize_constrained (L-BFGS-B)...")
    t0 = time.perf_counter()
    result = optimize_constrained(state)
    t_opt = time.perf_counter() - t0

    print(f"  Time:      {t_opt:.3f}s")
    print(f"  Objective: {result['objective']:.8e}")
    print(f"  Converged: {result['success']}")
    scipy_res = result["scipy_result"]
    print(f"  Iterations: {scipy_res.nit}  Func evals: {scipy_res.nfev}")
    print(f"  Total charge: {np.sum(result['qstore']):.6f}")

    # --- Per-molecule breakdown ---
    print(f"\n{'=' * 70}")
    print("  Per-Molecule Charges")
    print(f"{'=' * 70}")

    offset = 0
    for name, mol, record, labels in all_data:
        n = mol.n_atoms
        q_mol = result["qstore"][offset:offset + n]
        print(f"\n  {name} (charge: {np.sum(q_mol):+.6f})")
        print(f"    {'Atom':>4} {'Elem':>4} {'Label':<16} {'Charge':>12}")
        print(f"    {'-' * 38}")
        for i in range(n):
            elem = SYMBOLS[mol.atoms[i].atomic_number]
            print(f"    {i:>4} {elem:>4} {labels[i]:<16} {q_mol[i]:>+12.6f}")
        offset += n

    # --- Transferable charge summary ---
    print(f"\n{'=' * 70}")
    print("  Transferable Charges (by atom type)")
    print(f"{'=' * 70}")

    type_charges = {}
    for i, lbl in enumerate(all_labels):
        type_charges.setdefault(lbl, []).append(result["qstore"][i])

    print(f"  {'Label':<16} {'Charge':>12} {'Count':>6} {'Max diff':>12}")
    print(f"  {'-' * 48}")
    all_constraints_ok = True
    for lbl in sorted(type_charges.keys()):
        charges = type_charges[lbl]
        q = charges[0]
        max_diff = max(charges) - min(charges)
        ok = max_diff < 1e-10
        if not ok:
            all_constraints_ok = False
        status = "" if ok else " *** FAIL ***"
        print(f"  {lbl:<16} {q:>+12.6f} {len(charges):>6} {max_diff:>12.2e}{status}")

    print(f"\n  Constraints: {'ALL PASS' if all_constraints_ok else 'SOME FAILED'}")
    return all_constraints_ok


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
