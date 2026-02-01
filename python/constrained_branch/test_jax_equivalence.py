"""Verify numpy and JAX backends produce identical results for ethanol.

Tests two things:
1. Function equivalence: kaisq/dkaisq produce identical values at the same input.
2. Optimization equivalence: both backends converge to the same charges.

Note on JAX performance: The current implementation calls JAX ops individually
inside Python for-loops, incurring ~30x dispatch overhead per call vs numpy.
JAX speedups require JIT-compiling entire functions (eliminating Python loops
via jax.vmap/jax.jit), which is a future step.
"""

import time

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord

from openff_pympfit.mpfit._legacy_constrained import (
    setup_from_gdma_records as setup_np,
    kaisq as kaisq_np,
    dkaisq as dkaisq_np,
    optimize_constrained as optimize_np,
    count_parameters,
    generate_atom_type_labels_from_symmetry,
)
from openff_pympfit.mpfit._legacy_constrained_jax import (
    setup_from_gdma_records as setup_jax,
    kaisq as kaisq_jax,
    dkaisq as dkaisq_jax,
    optimize_constrained as optimize_jax,
)


def make_ethanol_record():
    """Generate GDMA record for ethanol."""
    molecule = Molecule.from_smiles("CCO", allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    gdma_settings = GDMASettings()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=True
    )
    gdma_record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings
    )
    return molecule, gdma_record


def test_function_equivalence(gdma_record, labels, jnp):
    """Test that kaisq and dkaisq produce identical values at fixed inputs."""
    print("\n--- Function Equivalence ---")

    state_np = setup_np(gdma_record, labels)
    state_jax = setup_jax(gdma_record, labels)
    n_params = count_parameters(state_np)

    all_passed = True
    for seed, scale, name in [(42, 0.01, "small p0"), (7, 1.0, "large p0"), (None, 0.0, "zero p0")]:
        if seed is not None:
            np.random.seed(seed)
            p0 = np.random.randn(n_params) * scale
        else:
            p0 = np.zeros(n_params)

        # Reset states
        state_np = setup_np(gdma_record, labels)
        state_jax = setup_jax(gdma_record, labels)

        obj_np = kaisq_np(p0, state_np)
        obj_jax = kaisq_jax(p0, state_jax, xp=jnp)
        obj_match = np.isclose(obj_np, obj_jax, rtol=1e-12)

        grad_np = dkaisq_np(p0, state_np)
        grad_jax = dkaisq_jax(p0, state_jax, xp=jnp)
        grad_max_diff = np.max(np.abs(grad_np - grad_jax))
        grad_match = np.allclose(grad_np, grad_jax, rtol=1e-12)

        passed = obj_match and grad_match
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: kaisq match={obj_match}, dkaisq max_diff={grad_max_diff:.2e} [{status}]")

    return all_passed


def test_optimization_equivalence(gdma_record, labels, jnp):
    """Test that both backends converge to the same charges."""
    print("\n--- Optimization Equivalence ---")

    state_np = setup_np(gdma_record, labels)
    state_np.molecule_charge = 0.0
    t0 = time.perf_counter()
    result_np = optimize_np(state_np)
    t_np = time.perf_counter() - t0

    state_jax = setup_jax(gdma_record, labels)
    state_jax.molecule_charge = 0.0
    t0 = time.perf_counter()
    result_jax = optimize_jax(state_jax, xp=jnp)
    t_jax = time.perf_counter() - t0

    q_np = result_np["qstore"]
    q_jax = result_jax["qstore"]

    print(f"\n  {'Atom':<6} {'numpy':>14} {'jax':>14} {'diff':>12}")
    print(f"  {'-'*48}")
    for i, (qn, qj) in enumerate(zip(q_np, q_jax)):
        diff = abs(qn - qj)
        print(f"  {i:<6} {qn:>+14.10f} {qj:>+14.10f} {diff:>12.2e}")

    max_charge_diff = np.max(np.abs(q_np - q_jax))
    obj_np = result_np["objective"]
    obj_jax = result_jax["objective"]
    obj_rel_diff = abs(obj_np - obj_jax) / max(abs(obj_np), 1e-15)

    # Charges may differ due to optimizer path divergence from floating-point
    # differences between backends. Check that objectives are close (both found
    # equally good solutions) and charges are reasonably similar.
    charges_close = np.allclose(q_np, q_jax, atol=1e-3)
    objectives_close = np.isclose(obj_np, obj_jax, rtol=1e-4)
    passed = charges_close and objectives_close

    print(f"\n  Max charge diff:  {max_charge_diff:.2e}")
    print(f"  Objective numpy:  {obj_np:.15e}")
    print(f"  Objective jax:    {obj_jax:.15e}")
    print(f"  Objective rel diff: {obj_rel_diff:.2e}")
    print(f"\n  Time numpy: {t_np:.3f}s")
    print(f"  Time jax:   {t_jax:.3f}s  ({t_jax/t_np:.1f}x slower)")
    print(f"  Note: JAX overhead is from per-op dispatch in Python loops.")
    print(f"        Speedups require jax.jit over vectorized functions.")
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    print("=" * 60)
    print("  JAX Equivalence Test — Ethanol")
    print("=" * 60)

    molecule, gdma_record = make_ethanol_record()
    labels = generate_atom_type_labels_from_symmetry(molecule)

    func_pass = test_function_equivalence(gdma_record, labels, jnp)
    opt_pass = test_optimization_equivalence(gdma_record, labels, jnp)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Function equivalence: {'PASS' if func_pass else 'FAIL'}")
    print(f"  Optimization equivalence: {'PASS' if opt_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if (func_pass and opt_pass) else 'FAIL'}")

    return func_pass and opt_pass


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
