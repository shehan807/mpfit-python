"""Compare original and refactored implementations on N4444 + P4444.

Tests that:
1. Original numpy-vectorized and original JIT-JAX produce the same charges.
2. Refactored (xp=np) and refactored (xp=jnp) produce the same charges.
3. Original and refactored produce the same charges.
4. Reports timing for all four approaches.
"""

import time

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units.elements import SYMBOLS
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord

# Original implementation
from openff_pympfit.mpfit._legacy_constrained_original import (
    setup_from_multiple_gdma_records as setup_original,
    optimize_constrained as optimize_original_np,
    optimize_constrained_jit as optimize_original_jit,
    generate_quaternary_ammonium_atom_types,
    count_parameters,
    kaisq_vectorized as kaisq_original,
    dkaisq_vectorized as dkaisq_original,
    kaisq_jit as kaisq_jit_original,
    dkaisq_jit as dkaisq_jit_original,
    _make_jit_functions,
)

# Refactored implementation
from openff_pympfit.mpfit._legacy_constrained_jax import (
    setup_from_gdma_records as setup_refactored,
    optimize_constrained as optimize_refactored,
    kaisq as kaisq_refactored,
    dkaisq as dkaisq_refactored,
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


def make_n4444_p4444_records():
    """Generate GDMA records for N4444 and P4444."""
    gdma_settings = GDMASettings()
    records = []
    all_labels = []
    molecules = []

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
        molecules.append(mol)
        labels = assign_quaternary_labels(mol, center_label)
        all_labels.extend(labels)
        print(f"    {mol.n_atoms} atoms")

    return records, all_labels, molecules


def print_charge_comparison(labels, charges_dict, mol_boundary):
    """Print side-by-side charge comparison."""
    names = list(charges_dict.keys())
    header = f"  {'Atom':<6} {'Label':>10}"
    for name in names:
        header += f" {name:>14}"
    header += f" {'max_diff':>10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for i in range(len(labels)):
        line = f"  {i:<6} {labels[i]:>10}"
        vals = [charges_dict[name][i] for name in names]
        for v in vals:
            line += f" {v:>+14.8f}"
        max_diff = max(vals) - min(vals)
        line += f" {max_diff:>10.2e}"
        if i == mol_boundary - 1:
            line += "  <-- N4444/P4444 boundary"
        print(line)


def test_function_equivalence(records, all_labels, jnp):
    """Test kaisq/dkaisq equivalence at fixed inputs across all implementations."""
    print("\n--- Function Equivalence (fixed p0) ---")

    state_orig = setup_original(records, all_labels)
    state_orig.molecule_charge = 2.0
    state_refac = setup_refactored(records, all_labels)
    state_refac.molecule_charge = 2.0

    n_params = count_parameters(state_orig)
    np.random.seed(42)
    p0 = np.random.randn(n_params) * 0.01

    # Original numpy
    state_orig_copy = setup_original(records, all_labels)
    state_orig_copy.molecule_charge = 2.0
    obj_orig = kaisq_original(p0, state_orig_copy)
    grad_orig = dkaisq_original(p0, state_orig_copy)

    # Original JIT
    state_jit = setup_original(records, all_labels)
    state_jit.molecule_charge = 2.0
    jit_fns = _make_jit_functions(state_jit)
    obj_jit = kaisq_jit_original(p0, state_jit, jit_fns)
    grad_jit = dkaisq_jit_original(p0, state_jit, jit_fns)

    # Refactored xp=np
    state_ref_np = setup_refactored(records, all_labels)
    state_ref_np.molecule_charge = 2.0
    obj_ref_np = kaisq_refactored(p0, state_ref_np, xp=np)
    grad_ref_np = dkaisq_refactored(p0, state_ref_np, xp=np)

    # Refactored xp=jnp
    state_ref_jax = setup_refactored(records, all_labels)
    state_ref_jax.molecule_charge = 2.0
    obj_ref_jax = kaisq_refactored(p0, state_ref_jax, xp=jnp)
    grad_ref_jax = dkaisq_refactored(p0, state_ref_jax, xp=jnp)

    results = {
        "orig_np": (obj_orig, grad_orig),
        "orig_jit": (obj_jit, grad_jit),
        "refac_np": (obj_ref_np, grad_ref_np),
        "refac_jax": (obj_ref_jax, grad_ref_jax),
    }

    all_passed = True
    ref_obj, ref_grad = results["orig_np"]
    for name, (obj, grad) in results.items():
        obj_match = np.isclose(obj, ref_obj, rtol=1e-10)
        grad_max_diff = np.max(np.abs(grad - ref_grad))
        grad_match = np.allclose(grad, ref_grad, rtol=1e-10)
        passed = obj_match and grad_match
        all_passed = all_passed and passed
        print(f"  {name:>10}: kaisq={obj:.12e}  grad_max_diff={grad_max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    return all_passed


def test_optimization_equivalence(records, all_labels, molecules, jnp):
    """Run all four optimization approaches and compare."""
    print("\n--- Optimization Comparison (N4444 + P4444) ---")

    total_charge = 2.0
    n4_atoms = molecules[0].n_atoms
    total_atoms = sum(m.n_atoms for m in molecules)

    results = {}

    # 1. Original numpy-vectorized
    state = setup_original(records, all_labels)
    state.molecule_charge = total_charge
    t0 = time.perf_counter()
    res = optimize_original_np(state, verbose=False)
    t = time.perf_counter() - t0
    results["orig_np"] = {"qstore": res["qstore"], "objective": res["objective"], "time": t}

    # 2. Original JIT
    state = setup_original(records, all_labels)
    state.molecule_charge = total_charge
    t0 = time.perf_counter()
    res = optimize_original_jit(state, verbose=False)
    t = time.perf_counter() - t0
    results["orig_jit"] = {"qstore": res["qstore"], "objective": res["objective"], "time": t}

    # 3. Refactored xp=np
    state = setup_refactored(records, all_labels)
    state.molecule_charge = total_charge
    t0 = time.perf_counter()
    res = optimize_refactored(state, xp=np)
    t = time.perf_counter() - t0
    results["refac_np"] = {"qstore": res["qstore"], "objective": res["objective"], "time": t}

    # 4. Refactored xp=jnp
    state = setup_refactored(records, all_labels)
    state.molecule_charge = total_charge
    t0 = time.perf_counter()
    res = optimize_refactored(state, xp=jnp)
    t = time.perf_counter() - t0
    results["refac_jax"] = {"qstore": res["qstore"], "objective": res["objective"], "time": t}

    # Timing summary
    print(f"\n  {'Method':>10} {'Objective':>20} {'Time':>10} {'Speedup':>10}")
    print(f"  {'-' * 54}")
    ref_time = results["orig_np"]["time"]
    for name, res in results.items():
        speedup = ref_time / res["time"]
        print(f"  {name:>10} {res['objective']:>20.12e} {res['time']:>9.2f}s {speedup:>9.2f}x")

    # Charge comparison
    print(f"\n  Charge comparison (vs orig_np):")
    ref_q = results["orig_np"]["qstore"]
    all_passed = True
    for name, res in results.items():
        max_diff = np.max(np.abs(res["qstore"] - ref_q))
        obj_rel = abs(res["objective"] - results["orig_np"]["objective"]) / abs(results["orig_np"]["objective"])
        close = np.allclose(res["qstore"], ref_q, atol=1e-3) and obj_rel < 1e-4
        all_passed = all_passed and close
        print(f"  {name:>10}: max_charge_diff={max_diff:.2e}  obj_rel_diff={obj_rel:.2e}  [{'PASS' if close else 'FAIL'}]")

    # Detailed charge table
    print(f"\n  Full charge table:")
    print_charge_comparison(
        all_labels,
        {name: res["qstore"] for name, res in results.items()},
        n4_atoms,
    )

    return all_passed


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    print("=" * 70)
    print("  JAX Equivalence Test — N4444 + P4444 (106 atoms)")
    print("=" * 70)

    print("\n  Generating GDMA records (Psi4)...")
    records, all_labels, molecules = make_n4444_p4444_records()

    func_pass = test_function_equivalence(records, all_labels, jnp)
    opt_pass = test_optimization_equivalence(records, all_labels, molecules, jnp)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Function equivalence:     {'PASS' if func_pass else 'FAIL'}")
    print(f"  Optimization equivalence: {'PASS' if opt_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if (func_pass and opt_pass) else 'FAIL'}")

    return func_pass and opt_pass


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
