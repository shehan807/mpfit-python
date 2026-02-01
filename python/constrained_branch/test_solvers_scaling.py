"""Compare SciPySolver and JAXSolver on replicated ethanol molecules.

Duplicates a single molecule 10-20 times with shared atom-type labels to
create a large constraint matrix, testing where JAX JIT outperforms numpy.
Reports compilation vs optimization time separately.
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
    generate_atom_type_labels_from_symmetry,
    SciPySolver,
    JAXSolver,
    count_parameters,
)


def make_ethanol_record():
    """Generate a single GDMA record for ethanol."""
    molecule = Molecule.from_smiles("CCO", allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    gdma_settings = GDMASettings()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=True,
    )
    record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings,
    )
    labels = generate_atom_type_labels_from_symmetry(molecule)
    return molecule, record, labels


def run_comparison(n_copies, record, labels_single):
    """Run SciPySolver vs JAXSolver for n_copies of ethanol."""
    records = [record] * n_copies
    all_labels = labels_single * n_copies
    n_atoms_total = len(all_labels)

    state = setup_from_gdma_records(records, all_labels)
    state.molecule_charge = 0.0
    n_params = count_parameters(state)

    # --- SciPySolver ---
    t0 = time.perf_counter()
    solver_np = SciPySolver(state)
    t_setup_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_np = solver_np.optimize()
    t_opt_np = time.perf_counter() - t0

    # --- JAXSolver ---
    t0 = time.perf_counter()
    solver_jax = JAXSolver(state)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_jax = solver_jax.optimize()
    t_opt_jax = time.perf_counter() - t0

    # --- Check ---
    max_charge_diff = np.max(np.abs(res_np["qstore"] - res_jax["qstore"]))
    obj_rel_diff = abs(res_np["objective"] - res_jax["objective"]) / max(abs(res_np["objective"]), 1e-15)
    charges_close = np.allclose(res_np["qstore"], res_jax["qstore"], atol=1e-3)
    obj_close = obj_rel_diff < 1e-4
    passed = charges_close and obj_close

    speedup = t_opt_np / t_opt_jax if t_opt_jax > 0 else float("inf")

    return {
        "n_copies": n_copies,
        "n_atoms": n_atoms_total,
        "n_params": n_params,
        "t_setup_np": t_setup_np,
        "t_opt_np": t_opt_np,
        "t_compile_jax": t_compile,
        "t_opt_jax": t_opt_jax,
        "obj_np": res_np["objective"],
        "obj_jax": res_jax["objective"],
        "qstore_np": res_np["qstore"],
        "qstore_jax": res_jax["qstore"],
        "labels": all_labels,
        "max_charge_diff": max_charge_diff,
        "speedup": speedup,
        "passed": passed,
    }


def main():
    print("=" * 70)
    print("  Solver Scaling Test — Replicated Ethanol")
    print("=" * 70)

    print("\n  Generating ethanol GDMA record (Psi4)...")
    molecule, record, labels_single = make_ethanol_record()
    print(f"  Single molecule: {molecule.n_atoms} atoms, labels: {labels_single}")

    copy_counts = [1, 2, 5, 10, 15, 20]
    results = []

    for n in copy_counts:
        print(f"\n--- {n} copies ({n * molecule.n_atoms} atoms) ---")
        res = run_comparison(n, record, labels_single)
        results.append(res)
        print(f"  Params:       {res['n_params']}")
        print(f"  SciPy setup:  {res['t_setup_np']:.3f}s  opt: {res['t_opt_np']:.3f}s")
        print(f"  JAX compile:  {res['t_compile_jax']:.3f}s  opt: {res['t_opt_jax']:.3f}s")
        print(f"  Speedup (opt): {res['speedup']:.2f}x")
        print(f"  Obj SciPy:    {res['obj_np']:.8e}")
        print(f"  Obj JAX:      {res['obj_jax']:.8e}")
        print(f"  Obj rel diff: {abs(res['obj_np'] - res['obj_jax']) / max(abs(res['obj_np']), 1e-15):.2e}")
        print(f"  Charge diff:  {res['max_charge_diff']:.2e}  [{'PASS' if res['passed'] else 'FAIL'}]")
        # Per-atom charge table (one row per unique label)
        seen = {}
        for i, lbl in enumerate(res["labels"]):
            if lbl not in seen:
                seen[lbl] = i
        print(f"  {'Label':<12} {'SciPy':>12} {'JAX':>12} {'Diff':>12}")
        print(f"  {'-' * 50}")
        for lbl, i in seen.items():
            q_np = res["qstore_np"][i]
            q_jax = res["qstore_jax"][i]
            print(f"  {lbl:<12} {q_np:>12.6f} {q_jax:>12.6f} {q_np - q_jax:>12.2e}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Copies':>6} {'Atoms':>6} {'Params':>7} {'SciPy opt':>10} {'JAX compile':>12} {'JAX opt':>10} {'Speedup':>8} {'Status':>6}")
    print(f"  {'-' * 68}")
    for r in results:
        print(f"  {r['n_copies']:>6} {r['n_atoms']:>6} {r['n_params']:>7}"
              f" {r['t_opt_np']:>9.3f}s {r['t_compile_jax']:>11.3f}s {r['t_opt_jax']:>9.3f}s"
              f" {r['speedup']:>7.2f}x {'PASS' if r['passed'] else 'FAIL':>6}")

    all_passed = all(r["passed"] for r in results)
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
