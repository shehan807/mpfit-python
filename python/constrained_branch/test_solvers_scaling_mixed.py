"""Compare SciPySolver and JAXSolver on mixed small molecules.

Uses a pool of ~50 distinct small molecules with symmetry-based atom type labels
to create realistic multi-molecule fitting problems.
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
    DirectSolver,
    count_parameters,
)

# Pool of small molecules (neutral, common organics)
SMILES_POOL = [
    "C",            # methane
    "CC",           # ethane
    "CCC",          # propane
    "CCCC",         # butane
    "CC(C)C",       # isobutane
    "CCO",          # ethanol
    "CCCO",         # 1-propanol
    "CC(C)O",       # 2-propanol
    "CO",           # methanol
    "C=C",          # ethylene
    "C=CC",         # propene
    "CC=CC",        # 2-butene
    "C#C",          # acetylene
    "C#CC",         # propyne
    "CC(=O)C",      # acetone
    "CC=O",         # acetaldehyde
    "C=O",          # formaldehyde
    "CC(=O)O",      # acetic acid
    "C(=O)O",       # formic acid
    "COC",          # dimethyl ether
    "CCOC",         # ethyl methyl ether
    "CCOCC",        # diethyl ether
    "CCN",          # ethylamine
    "CN",           # methylamine
    "CNN",          # methylhydrazine
    "N",            # ammonia
    "O",            # water
    "CC(C)(C)C",    # neopentane
    "CCCCC",        # pentane
    "CCC(C)C",      # isopentane
    "C1CC1",        # cyclopropane
    "C1CCC1",       # cyclobutane
    "C1CCCC1",      # cyclopentane
    "OO",           # hydrogen peroxide
    "CS",           # methanethiol
    "CCS",          # ethanethiol
    "CSC",          # dimethyl sulfide
    "CF",           # fluoromethane
    "CCF",          # fluoroethane
    "CCl",          # chloromethane
    "CCCl",         # chloroethane
    "C(F)(F)F",     # trifluoromethane
    "CC#N",         # acetonitrile
    "C#N",          # hydrogen cyanide
    "CCCC(=O)C",    # 2-pentanone
    "CCC=O",        # propanal
    "CCCCCC",       # hexane
    "CCCCCCC",      # heptane
    "CC(=O)OC",     # methyl acetate
    "COC=O",        # methyl formate
]


def validate_smiles():
    """Check all SMILES parse correctly and print atom counts."""
    print(f"  Validating {len(SMILES_POOL)} SMILES...")
    valid = []
    for smi in SMILES_POOL:
        try:
            mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
            valid.append((smi, mol.n_atoms))
        except Exception as e:
            print(f"  INVALID: {smi} — {e}")
    print(f"  {len(valid)}/{len(SMILES_POOL)} valid")
    return valid


def make_record(smiles):
    """Generate a GDMA record for a single molecule."""
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
    labels = generate_atom_type_labels_from_symmetry(molecule)
    return molecule, record, labels


def run_comparison(records, all_labels, total_charge=0.0):
    """Run SciPySolver vs DirectSolver (and optionally JAXSolver) for a set of molecules."""
    n_atoms_total = len(all_labels)

    state = setup_from_gdma_records(records, all_labels)
    state.molecule_charge = total_charge
    n_params = count_parameters(state)

    # --- SciPySolver (iterative L-BFGS-B) ---
    t0 = time.perf_counter()
    solver_np = SciPySolver(state)
    t_setup_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_np = solver_np.optimize()
    t_opt_np = time.perf_counter() - t0

    # --- DirectSolver (exact linear solve) ---
    t0 = time.perf_counter()
    solver_direct = DirectSolver(state)
    t_setup_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_direct = solver_direct.solve()
    t_solve_direct = time.perf_counter() - t0

    # --- JAXSolver (JIT + LBFGS) ---
    res_jax = None
    t_compile = None
    t_opt_jax = None
    try:
        t0 = time.perf_counter()
        solver_jax = JAXSolver(state)
        t_compile = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_jax = solver_jax.optimize()
        t_opt_jax = time.perf_counter() - t0
    except Exception as e:
        print(f"  JAXSolver failed: {e}")

    # --- Check: SciPy vs Direct ---
    max_charge_diff = np.max(np.abs(res_np["qstore"] - res_direct["qstore"]))
    obj_rel_diff = abs(res_np["objective"] - res_direct["objective"]) / max(abs(res_np["objective"]), 1e-15)
    charges_close = np.allclose(res_np["qstore"], res_direct["qstore"], atol=1e-3)
    obj_close = obj_rel_diff < 1e-4
    passed = charges_close and obj_close

    speedup = t_opt_np / t_solve_direct if t_solve_direct > 0 else float("inf")

    return {
        "n_atoms": n_atoms_total,
        "n_params": n_params,
        "t_setup_np": t_setup_np,
        "t_opt_np": t_opt_np,
        "t_setup_direct": t_setup_direct,
        "t_solve_direct": t_solve_direct,
        "t_compile_jax": t_compile,
        "t_opt_jax": t_opt_jax,
        "obj_np": res_np["objective"],
        "obj_direct": res_direct["objective"],
        "obj_jax": res_jax["objective"] if res_jax else None,
        "qstore_np": res_np["qstore"],
        "qstore_direct": res_direct["qstore"],
        "qstore_jax": res_jax["qstore"] if res_jax else None,
        "labels": all_labels,
        "max_charge_diff": max_charge_diff,
        "speedup": speedup,
        "passed": passed,
    }


def main():
    print("=" * 70)
    print("  Solver Scaling Test — Mixed Small Molecules")
    print("=" * 70)

    # Validate SMILES
    valid = validate_smiles()

    # Generate all GDMA records upfront
    print(f"\n  Generating GDMA records for {len(SMILES_POOL)} molecules (Psi4)...")
    all_data = []
    for i, smi in enumerate(SMILES_POOL):
        print(f"  [{i+1:2d}/{len(SMILES_POOL)}] {smi}...", end="", flush=True)
        try:
            mol, record, labels = make_record(smi)
            all_data.append((smi, mol, record, labels))
            print(f" {mol.n_atoms} atoms")
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"\n  Successfully generated {len(all_data)} records")

    # Test at different molecule counts
    mol_counts = [5, 10, 25, min(50, len(all_data))]

    for n_mols in mol_counts:
        if n_mols > len(all_data):
            print(f"\n--- Skipping {n_mols} molecules (only {len(all_data)} available) ---")
            continue

        subset = all_data[:n_mols]
        records = [d[2] for d in subset]
        all_labels = []
        for d in subset:
            all_labels.extend(d[3])

        n_atoms = len(all_labels)
        smiles_used = [d[0] for d in subset]

        print(f"\n{'=' * 70}")
        print(f"--- {n_mols} molecules ({n_atoms} atoms) ---")
        print(f"{'=' * 70}")
        print(f"  Molecules: {smiles_used}")

        res = run_comparison(records, all_labels)

        print(f"  Params:       {res['n_params']}")
        print(f"  SciPy setup:  {res['t_setup_np']:.3f}s  opt: {res['t_opt_np']:.3f}s")
        print(f"  Direct setup: {res['t_setup_direct']:.3f}s  solve: {res['t_solve_direct']:.3f}s")
        if res['t_compile_jax'] is not None:
            print(f"  JAX compile:  {res['t_compile_jax']:.3f}s  opt: {res['t_opt_jax']:.3f}s")
        print(f"  Speedup (Direct vs SciPy): {res['speedup']:.2f}x")
        print(f"  Obj SciPy:    {res['obj_np']:.8e}")
        print(f"  Obj Direct:   {res['obj_direct']:.8e}")
        if res['obj_jax'] is not None:
            print(f"  Obj JAX:      {res['obj_jax']:.8e}")
        print(f"  Obj rel diff (SciPy vs Direct): {abs(res['obj_np'] - res['obj_direct']) / max(abs(res['obj_np']), 1e-15):.2e}")
        print(f"  Charge diff (SciPy vs Direct): {res['max_charge_diff']:.2e}  [{'PASS' if res['passed'] else 'FAIL'}]")

        # Per-unique-label charge table
        seen = {}
        for i, lbl in enumerate(res["labels"]):
            if lbl not in seen:
                seen[lbl] = i
        header = f"  {'Label':<16} {'SciPy':>12} {'Direct':>12} {'Diff':>12}"
        if res['qstore_jax'] is not None:
            header += f" {'JAX':>12}"
        print(f"\n{header}")
        print(f"  {'-' * (54 + (14 if res['qstore_jax'] is not None else 0))}")
        for lbl, i in seen.items():
            q_np = res["qstore_np"][i]
            q_direct = res["qstore_direct"][i]
            line = f"  {lbl:<16} {q_np:>12.6f} {q_direct:>12.6f} {q_np - q_direct:>12.2e}"
            if res['qstore_jax'] is not None:
                line += f" {res['qstore_jax'][i]:>12.6f}"
            print(line)

    return True


if __name__ == "__main__":
    main()
