"""
Constrained MPFIT Implementation

Fits partial charges to reproduce GDMA multipoles with atom-type equivalence constraints.
Atoms with the same type label are constrained to have equal total charges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from openff_pympfit.mpfit.core import _regular_solid_harmonic

if TYPE_CHECKING:
    from openff_pympfit.gdma.storage import MoleculeGDMARecord


@dataclass
class ConstrainedMPFITState:
    """Container for constrained MPFIT state variables."""

    maxl: int = 4
    r1: float = 3.78
    r2: float = 9.45
    molecule_charge: float = 0.0
    conchg: float = 0.0

    atomtype: list[str] = field(default_factory=list)
    quse: np.ndarray | None = None
    allcharge: np.ndarray | None = None
    qstore: np.ndarray | None = None
    multipoles: np.ndarray | None = None
    xyzmult: np.ndarray | None = None
    xyzcharge: np.ndarray | None = None
    lmax: np.ndarray | None = None
    rvdw: np.ndarray | None = None


# =============================================================================
# Setup Functions
# =============================================================================


def setup_from_gdma_record(
    gdma_record: MoleculeGDMARecord,
    atom_type_labels: list[str],
) -> ConstrainedMPFITState:
    """Initialize state from a GDMA record."""
    from openff.toolkit import Molecule
    from openff.units import unit
    from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical

    molecule = Molecule.from_mapped_smiles(
        gdma_record.tagged_smiles, allow_undefined_stereo=True
    )
    n_atoms = molecule.n_atoms
    gdma_settings = gdma_record.gdma_settings

    state = ConstrainedMPFITState()
    state.r1 = gdma_settings.mpfit_inner_radius
    state.r2 = gdma_settings.mpfit_outer_radius
    state.atomtype = atom_type_labels

    conformer_bohr = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
    state.xyzcharge = conformer_bohr.copy()
    state.xyzmult = conformer_bohr.copy()
    state.rvdw = np.full(n_atoms, gdma_settings.mpfit_atom_radius)
    state.lmax = np.full(n_atoms, gdma_settings.limit, dtype=float)
    state.multipoles = _convert_flat_to_hierarchical(
        gdma_record.multipoles, n_atoms, gdma_settings.limit
    )
    state.quse = build_quse_matrix(state.xyzmult, state.xyzcharge, state.rvdw)
    state.allcharge = np.zeros((n_atoms, n_atoms))
    state.qstore = np.zeros(n_atoms)

    return state


def setup_from_multiple_gdma_records(
    gdma_records: list[MoleculeGDMARecord],
    atom_type_labels: list[str],
) -> ConstrainedMPFITState:
    """
    Initialize state from multiple GDMA records for transferable charge fitting.

    Stacks coordinates and multipoles from all molecules into a combined system.
    Atoms with matching labels across different molecules will be constrained
    to have the same total charge.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord]
        GDMA records for each molecule
    atom_type_labels : list[str]
        Combined atom type labels for all atoms across all molecules.
        Length must equal sum of atoms in all molecules.

    Returns
    -------
    ConstrainedMPFITState
        Combined state ready for optimization
    """
    from openff.toolkit import Molecule
    from openff.units import unit
    from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical

    # Collect data from each molecule
    all_xyz = []
    all_multipoles = []
    all_rvdw = []
    all_lmax = []
    total_atoms = 0

    for gdma_record in gdma_records:
        molecule = Molecule.from_mapped_smiles(
            gdma_record.tagged_smiles, allow_undefined_stereo=True
        )
        n_atoms = molecule.n_atoms
        total_atoms += n_atoms
        gdma_settings = gdma_record.gdma_settings

        conformer_bohr = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
        all_xyz.append(conformer_bohr)

        multipoles = _convert_flat_to_hierarchical(
            gdma_record.multipoles, n_atoms, gdma_settings.limit
        )
        all_multipoles.append(multipoles)

        all_rvdw.append(np.full(n_atoms, gdma_settings.mpfit_atom_radius))
        all_lmax.append(np.full(n_atoms, gdma_settings.limit, dtype=float))

    # Validate labels length
    if len(atom_type_labels) != total_atoms:
        raise ValueError(
            f"atom_type_labels has {len(atom_type_labels)} entries, "
            f"but total atoms across all molecules is {total_atoms}"
        )

    # Use settings from first record for r1/r2
    gdma_settings = gdma_records[0].gdma_settings

    state = ConstrainedMPFITState()
    state.r1 = gdma_settings.mpfit_inner_radius
    state.r2 = gdma_settings.mpfit_outer_radius
    state.atomtype = atom_type_labels

    # Stack arrays from all molecules
    state.xyzcharge = np.vstack(all_xyz)
    state.xyzmult = np.vstack(all_xyz)
    state.multipoles = np.vstack(all_multipoles)
    state.rvdw = np.concatenate(all_rvdw)
    state.lmax = np.concatenate(all_lmax)

    # Build quse for combined system
    state.quse = build_quse_matrix(state.xyzmult, state.xyzcharge, state.rvdw)
    state.allcharge = np.zeros((total_atoms, total_atoms))
    state.qstore = np.zeros(total_atoms)

    return state


def build_quse_matrix(
    xyzmult: np.ndarray,
    xyzcharge: np.ndarray,
    rvdw: np.ndarray,
) -> np.ndarray:
    """Build binary mask: quse[s,i]=1 if atom i affects site s."""
    n_sites = xyzmult.shape[0]
    n_atoms = xyzcharge.shape[0]
    quse = np.zeros((n_sites, n_atoms), dtype=int)

    for i in range(n_sites):
        for j in range(n_atoms):
            rqm = np.sqrt(np.sum((xyzmult[i] - xyzcharge[j]) ** 2))
            if rqm < rvdw[i]:
                quse[i, j] = 1

    return quse


def count_parameters(state: ConstrainedMPFITState) -> int:
    """Count free parameters after applying constraints."""
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = quse.shape[0]
    n_params = 0

    for i in range(n_atoms):
        n_sites_using = np.sum(quse[:, i])

        if i == 0:
            n_params += n_sites_using
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
            if twin is not None:
                n_params += n_sites_using - 1
            else:
                n_params += n_sites_using

    return n_params


# =============================================================================
# Core Algorithm Functions
# =============================================================================


def rsh(l: int, m: int, cs: int, xyz: np.ndarray) -> float:
    """Evaluate regular solid harmonic at point xyz (scalar version)."""
    from openff_pympfit.mpfit.core import _regular_solid_harmonic
    return _regular_solid_harmonic(l, m, cs, xyz[0], xyz[1], xyz[2])


def expandcharge(p0: np.ndarray, state: ConstrainedMPFITState) -> None:
    """Map reduced parameters to full charges with atom-type constraints."""
    atomtype = state.atomtype
    quse = state.quse
    n_sites = state.xyzmult.shape[0]
    n_atoms = len(atomtype)

    state.allcharge = np.zeros((n_sites, n_atoms))
    state.qstore = np.zeros(n_atoms)
    count = 0

    for i in range(n_atoms):
        charge_sum = 0.0

        if i == 0:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    state.allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
            state.qstore[i] = charge_sum
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)

            if twin is not None:
                count1 = np.sum(quse[:, i])
                count2 = 1
                for j in range(n_sites):
                    if quse[j, i] == 1 and count2 < count1:
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                        count2 += 1
                    elif quse[j, i] == 1 and count2 == count1:
                        state.allcharge[j, i] = state.qstore[twin] - charge_sum
                        state.qstore[i] = state.qstore[twin]
            else:
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                state.qstore[i] = charge_sum


def kaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> float:
    """Objective function: sum of squared multipole errors."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    n_atoms = n_sites
    maxl = state.maxl
    sumkai = 0.0

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        W = np.zeros(maxl + 1)
        for i in range(int(state.lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        site_sum = 0.0
        for l in range(int(state.lmax[s]) + 1):
            if l == 0:
                sum1 = sum(q0[j] * rsh(0, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
                          for j in range(n_atoms))
                site_sum = (4.0 * np.pi) * W[0] * (state.multipoles[s, 0, 0, 0] - sum1) ** 2
            else:
                for m in range(l + 1):
                    if m == 0:
                        sum1 = sum(q0[j] * rsh(l, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
                                  for j in range(n_atoms))
                        site_sum += (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, 0, 0] - sum1) ** 2
                    else:
                        for cs in [0, 1]:
                            sum1 = sum(q0[j] * rsh(l, m, cs, state.xyzcharge[j] - state.xyzmult[s])
                                      for j in range(n_atoms))
                            site_sum += (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, cs] - sum1) ** 2

        sumkai += site_sum

    sumchg = np.sum(state.qstore)
    sumcon = state.conchg * (sumchg - state.molecule_charge) ** 2
    return sumkai + sumcon


def kaisq_vectorized(p0: np.ndarray, state: ConstrainedMPFITState) -> float:
    """Objective function: sum of squared multipole errors (vectorized inner loop)."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    maxl = state.maxl
    sumkai = 0.0

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        # Precompute W for all l values at this site
        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        # Displacement vectors from site s to all charge sites
        dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
        dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
        dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

        site_sum = 0.0
        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    # Vectorized: RSH for all atoms at once, then dot product
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                    sum1 = np.dot(q0, rsh_vals)
                    site_sum += weight * (state.multipoles[s, l, m, cs] - sum1) ** 2

        sumkai += site_sum

    sumchg = np.sum(state.qstore)
    sumcon = state.conchg * (sumchg - state.molecule_charge) ** 2
    return sumkai + sumcon


def dkaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Gradient of kaisq with respect to reduced parameters."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    n_atoms = n_sites
    maxl = state.maxl
    dparam = np.zeros(n_sites * n_atoms)

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        W = np.zeros(maxl + 1)
        for i in range(int(state.lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        for l in range(int(state.lmax[s]) + 1):
            if l == 0:
                sum1 = sum(q0[j] * rsh(0, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
                          for j in range(n_atoms))
                coeff = 2.0 * (4.0 * np.pi) * W[0] * (state.multipoles[s, 0, 0, 0] - sum1)
                for j in range(n_atoms):
                    dparam[s * n_atoms + j] -= coeff * rsh(0, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
            else:
                for m in range(l + 1):
                    if m == 0:
                        sum1 = sum(q0[j] * rsh(l, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
                                  for j in range(n_atoms))
                        coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, 0, 0] - sum1)
                        for j in range(n_atoms):
                            dparam[s * n_atoms + j] -= coeff * rsh(l, 0, 0, state.xyzcharge[j] - state.xyzmult[s])
                    else:
                        for cs in [0, 1]:
                            sum1 = sum(q0[j] * rsh(l, m, cs, state.xyzcharge[j] - state.xyzmult[s])
                                      for j in range(n_atoms))
                            coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, cs] - sum1)
                            for j in range(n_atoms):
                                dparam[s * n_atoms + j] -= coeff * rsh(l, m, cs, state.xyzcharge[j] - state.xyzmult[s])

    # Reorder from [site, atom] to [atom, site]
    dparam1 = np.zeros(n_sites * n_atoms)
    for i in range(n_sites):
        for j in range(n_atoms):
            dparam1[j * n_sites + i] = dparam[i * n_atoms + j]

    sumchg = np.sum(state.qstore)
    dparam1 += state.conchg * 2.0 * (sumchg - state.molecule_charge)

    return createdkaisq(dparam1, state)


def dkaisq_vectorized(p0: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Gradient of kaisq with respect to reduced parameters (vectorized)."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    n_atoms = n_sites
    maxl = state.maxl
    dparam = np.zeros((n_sites, n_atoms))  # 2D array for cleaner indexing

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        # Displacement vectors from site s to all charge sites
        dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
        dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
        dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    # Vectorized RSH
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                    sum1 = np.dot(q0, rsh_vals)
                    coeff = 2.0 * weight * (state.multipoles[s, l, m, cs] - sum1)
                    # Vectorized gradient update
                    dparam[s, :] -= coeff * rsh_vals

    # Reorder from [site, atom] to [atom, site] using transpose
    dparam1 = dparam.T.flatten()

    sumchg = np.sum(state.qstore)
    dparam1 += state.conchg * 2.0 * (sumchg - state.molecule_charge)

    return createdkaisq(dparam1, state)


def createdkaisq(dparam1: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Apply chain rule for constraint gradient transformation."""
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = n_atoms
    dparam1 = dparam1.copy()

    # Pass 1: Propagate constraint dependencies
    for i in range(1, n_atoms):
        twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
        if twin is not None:
            count1 = np.sum(quse[:, i])
            count2 = 1
            for j in range(n_sites):
                if quse[j, i] == 1 and count2 < count1:
                    count2 += 1
                elif quse[j, i] == 1 and count2 == count1:
                    for k in range(n_sites):
                        dparam1[twin * n_sites + k] += dparam1[i * n_sites + j]
                    for k in range(j):
                        dparam1[i * n_sites + k] -= dparam1[i * n_sites + j]

    # Pass 2: Extract free parameters
    n_params = count_parameters(state)
    dkaisq_out = np.zeros(n_params)
    count = 0

    for i in range(n_atoms):
        if i == 0:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    dkaisq_out[count] = dparam1[i * n_sites + j]
                    count += 1
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
            if twin is not None:
                count1 = np.sum(quse[:, i])
                count2 = 1
                for j in range(n_sites):
                    if quse[j, i] == 1 and count2 < count1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count2 += 1
                        count += 1
            else:
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count += 1

    return dkaisq_out


# =============================================================================
# JAX JIT
# =============================================================================


def _regular_solid_harmonic_jax(l, m, cs, x, y, z):
    """
    JAX-compatible vectorized regular solid harmonic.

    Uses jax.scipy.special.sph_harm_y for l < 4, explicit Cartesian
    formulas for l = 4. Works inside @jax.jit.
    """
    import jax.numpy as jnp
    from jax.scipy.special import sph_harm_y

    r = jnp.sqrt(x * x + y * y + z * z)
    safe_r = jnp.where(r > 1e-10, r, 1.0)

    if l == 4:
        # Explicit l=4 hexadecapole formulas (Cartesian, no trig)
        if m == 0:
            val = 0.125 * (8.0*z**4 - 24.0*(x**2 + y**2)*z**2
                           + 3.0*(x**4 + 2.0*x**2*y**2 + y**4))
        elif m == 1 and cs == 0:
            val = 0.25 * jnp.sqrt(10.0) * (4.0*x*z**3 - 3.0*x*z*(x**2 + y**2))
        elif m == 1 and cs == 1:
            val = 0.25 * jnp.sqrt(10.0) * (4.0*y*z**3 - 3.0*y*z*(x**2 + y**2))
        elif m == 2 and cs == 0:
            val = 0.25 * jnp.sqrt(5.0) * (x**2 - y**2) * (6.0*z**2 - x**2 - y**2)
        elif m == 2 and cs == 1:
            val = 0.25 * jnp.sqrt(5.0) * x * y * (6.0*z**2 - x**2 - y**2)
        elif m == 3 and cs == 0:
            val = 0.25 * jnp.sqrt(70.0) * z * (x**3 - 3.0*x*y**2)
        elif m == 3 and cs == 1:
            val = 0.25 * jnp.sqrt(70.0) * z * (3.0*x**2*y - y**3)
        elif m == 4 and cs == 0:
            val = 0.125 * jnp.sqrt(35.0) * (x**4 - 6.0*x**2*y**2 + y**4)
        elif m == 4 and cs == 1:
            val = 0.125 * jnp.sqrt(35.0) * x * y * (x**2 - y**2)
        else:
            val = jnp.zeros_like(r)
        return jnp.where(r > 1e-10, val, 0.0)

    # l < 4: use jax.scipy.special.sph_harm_y
    theta = jnp.arccos(z / safe_r)
    phi = jnp.arctan2(y, x)

    n_arr = jnp.full_like(theta, l, dtype=jnp.int32)
    m_arr = jnp.full_like(theta, m, dtype=jnp.int32)
    Y = sph_harm_y(n_arr, m_arr, theta, phi, n_max=l)

    norm = jnp.sqrt(4.0 * jnp.pi / (2.0 * l + 1.0))

    if m == 0:
        result = norm * safe_r**l * Y.real
    else:
        factor = jnp.sqrt(2.0) * ((-1.0) ** m) * norm * safe_r**l
        if cs == 0:
            result = factor * Y.real
        else:
            result = factor * Y.imag

    # Handle r ≈ 0: RSH = 1 only for (l=0, m=0, cs=0), else 0
    if l == 0 and m == 0 and cs == 0:
        return jnp.where(r > 1e-10, result, 1.0)
    else:
        return jnp.where(r > 1e-10, result, 0.0)


def _make_jit_functions(state: ConstrainedMPFITState):
    """
    Create JIT-compiled kaisq and dkaisq functions.

    Separates the pure computation (JIT-friendly) from the constraint
    logic (expandcharge, which stays in numpy).

    Uses _regular_solid_harmonic_jax to compute RSH values inline
    rather than looking up a precomputed cache.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    multipoles = jnp.array(state.multipoles, dtype=jnp.float64)
    xyzmult = jnp.array(state.xyzmult, dtype=jnp.float64)
    xyzcharge = jnp.array(state.xyzcharge, dtype=jnp.float64)
    rvdw = jnp.array(state.rvdw, dtype=jnp.float64)
    lmax_arr = jnp.array(state.lmax, dtype=jnp.int32)
    maxl = state.maxl
    n_sites = state.xyzmult.shape[0]
    r1 = state.r1
    r2 = state.r2

    # Build a list of (l, m, cs) component indices to iterate over
    components = []
    for l in range(maxl + 1):
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                components.append((l, m, cs))

    @jax.jit
    def kaisq_inner(allcharge_jnp, qstore_jnp, conchg, molecule_charge):
        """Pure function: compute objective from allcharge (no geometry)."""
        sumkai = 0.0

        for s in range(n_sites):
            q0 = allcharge_jnp[s, :]
            rmax = rvdw[s] + r2
            rminn = rvdw[s] + r1
            dx = xyzcharge[:, 0] - xyzmult[s, 0]
            dy = xyzcharge[:, 1] - xyzmult[s, 1]
            dz = xyzcharge[:, 2] - xyzmult[s, 2]

            for l, m, cs in components:
                W_l = (1.0 / (1.0 - 2.0 * l)) * (rmax ** (1 - 2 * l) - rminn ** (1 - 2 * l))
                angular = (4.0 * jnp.pi) if l == 0 else (4.0 * jnp.pi / (2.0 * l + 1.0))
                weight = angular * W_l

                rsh_vals = _regular_solid_harmonic_jax(l, m, cs, dx, dy, dz)
                sum1 = jnp.dot(q0, rsh_vals)
                sumkai += weight * (multipoles[s, l, m, cs] - sum1) ** 2

        sumchg = jnp.sum(qstore_jnp)
        sumcon = conchg * (sumchg - molecule_charge) ** 2
        return sumkai + sumcon

    @jax.jit
    def dkaisq_inner(allcharge_jnp):
        """Pure function: compute gradient w.r.t. allcharge (no geometry)."""
        n_atoms = allcharge_jnp.shape[1]
        dparam = jnp.zeros((n_sites, n_atoms))

        for s in range(n_sites):
            q0 = allcharge_jnp[s, :]
            rmax = rvdw[s] + r2
            rminn = rvdw[s] + r1
            dx = xyzcharge[:, 0] - xyzmult[s, 0]
            dy = xyzcharge[:, 1] - xyzmult[s, 1]
            dz = xyzcharge[:, 2] - xyzmult[s, 2]

            for l, m, cs in components:
                W_l = (1.0 / (1.0 - 2.0 * l)) * (rmax ** (1 - 2 * l) - rminn ** (1 - 2 * l))
                angular = (4.0 * jnp.pi) if l == 0 else (4.0 * jnp.pi / (2.0 * l + 1.0))
                weight = angular * W_l

                rsh_vals = _regular_solid_harmonic_jax(l, m, cs, dx, dy, dz)
                sum1 = jnp.dot(q0, rsh_vals)
                coeff = 2.0 * weight * (multipoles[s, l, m, cs] - sum1)
                dparam = dparam.at[s, :].add(-coeff * rsh_vals)

        return dparam

    return kaisq_inner, dkaisq_inner


def kaisq_jit(p0: np.ndarray, state: ConstrainedMPFITState, jit_fns: tuple) -> float:
    """Objective function using JIT-compiled inner computation."""
    import jax.numpy as jnp

    expandcharge(p0, state)
    kaisq_inner, _ = jit_fns

    result = kaisq_inner(
        jnp.array(state.allcharge, dtype=jnp.float64),
        jnp.array(state.qstore, dtype=jnp.float64),
        state.conchg,
        state.molecule_charge,
    )
    return float(result)


def dkaisq_jit(p0: np.ndarray, state: ConstrainedMPFITState, jit_fns: tuple) -> np.ndarray:
    """Gradient using JIT-compiled inner computation."""
    import jax.numpy as jnp

    expandcharge(p0, state)
    _, dkaisq_inner = jit_fns

    dparam = dkaisq_inner(jnp.array(state.allcharge, dtype=jnp.float64))

    # Convert back to numpy, reorder [site, atom] → [atom, site], flatten
    dparam_np = np.array(dparam)
    dparam1 = dparam_np.T.flatten()

    # Add charge constraint gradient
    sumchg = np.sum(state.qstore)
    dparam1 += state.conchg * 2.0 * (sumchg - state.molecule_charge)

    return createdkaisq(dparam1, state)


# =============================================================================
# Optimization
# =============================================================================


def optimize_constrained(
    state: ConstrainedMPFITState,
    p0_init: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Run constrained MPFIT optimization."""
    import time

    n_params = count_parameters(state)
    if p0_init is None:
        p0_init = np.zeros(n_params)

    if verbose:
        print(f"Starting optimization with {n_params} free parameters")
        print(f"Initial objective: {kaisq_vectorized(p0_init, state):.6e}")

    t_start = time.time()
    result = minimize(
        fun=lambda p: kaisq_vectorized(p, state),
        x0=p0_init,
        jac=lambda p: dkaisq_vectorized(p, state),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-4},
    )
    opt_time = time.time() - t_start

    if verbose:
        print(f"\nOptimization finished: {result.message}")
        print(f"Final objective: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")
        print(f"Optimization time: {opt_time:.1f}s ({opt_time/60:.2f} min)")

    expandcharge(result.x, state)

    return {
        "qstore": state.qstore.copy(),
        "allcharge": state.allcharge.copy(),
        "objective": result.fun,
        "success": result.success,
        "scipy_result": result,
    }


def optimize_constrained_jit(
    state: ConstrainedMPFITState,
    p0_init: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Run constrained MPFIT optimization with JAX JIT-compiled functions."""
    import time

    n_params = count_parameters(state)
    if p0_init is None:
        p0_init = np.zeros(n_params)

    # Create JIT functions
    if verbose:
        print("Creating JIT functions...")
    t_jit = time.time()
    jit_fns = _make_jit_functions(state)

    # Warm up JIT (first call triggers compilation)
    _ = kaisq_jit(p0_init, state, jit_fns)
    _ = dkaisq_jit(p0_init, state, jit_fns)
    t_jit = time.time() - t_jit

    if verbose:
        print(f"JIT compiled in {t_jit:.2f}s")
        print(f"Starting optimization with {n_params} free parameters")
        print(f"Initial objective: {kaisq_jit(p0_init, state, jit_fns):.6e}")

    t_start = time.time()
    result = minimize(
        fun=lambda p: kaisq_jit(p, state, jit_fns),
        x0=p0_init,
        jac=lambda p: dkaisq_jit(p, state, jit_fns),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-4},
    )
    opt_time = time.time() - t_start

    if verbose:
        print(f"\nOptimization finished: {result.message}")
        print(f"Final objective: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")
        print(f"Optimization time: {opt_time:.1f}s ({opt_time/60:.2f} min)")
        print(f"  (JIT compile: {t_jit:.1f}s)")

    expandcharge(result.x, state)

    return {
        "qstore": state.qstore.copy(),
        "allcharge": state.allcharge.copy(),
        "objective": result.fun,
        "success": result.success,
        "scipy_result": result,
    }


def verify_gradient(
    p0: np.ndarray,
    state: ConstrainedMPFITState,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Verify analytical gradient against numerical finite differences."""
    analytical = dkaisq(p0, state)
    numerical = np.zeros_like(p0)

    for i in range(len(p0)):
        p_plus, p_minus = p0.copy(), p0.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        numerical[i] = (kaisq(p_plus, state) - kaisq(p_minus, state)) / (2 * eps)

    max_rel_error = max(
        abs(analytical[i] - numerical[i]) / abs(numerical[i])
        for i in range(len(p0)) if abs(numerical[i]) > 1e-10
    )
    return analytical, numerical, max_rel_error


# =============================================================================
# High-Level API
# =============================================================================


def generate_atom_type_labels_from_symmetry(
    molecule,
    equivalize_hydrogens: bool = True,
    equivalize_other_atoms: bool = True,
) -> list[str]:
    """Generate atom type labels based on molecular symmetry."""
    from openff.recharge.utilities.toolkits import get_atom_symmetries
    from openff.units.elements import SYMBOLS

    symmetry_groups = get_atom_symmetries(molecule)
    labels = []

    for i, (atom, group) in enumerate(zip(molecule.atoms, symmetry_groups)):
        element = SYMBOLS[atom.atomic_number]
        is_hydrogen = atom.atomic_number == 1

        if (is_hydrogen and equivalize_hydrogens) or (not is_hydrogen and equivalize_other_atoms):
            labels.append(f"{element}{group}")
        else:
            labels.append(f"{element}_{i}")

    return labels


def fit_constrained_mpfit(
    gdma_record: MoleculeGDMARecord,
    atom_type_labels: list[str] | None = None,
    equivalize_hydrogens: bool = True,
    equivalize_other_atoms: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """High-level API for constrained MPFIT fitting."""
    from openff.toolkit import Molecule

    molecule = Molecule.from_mapped_smiles(
        gdma_record.tagged_smiles, allow_undefined_stereo=True
    )

    if atom_type_labels is None:
        atom_type_labels = generate_atom_type_labels_from_symmetry(
            molecule,
            equivalize_hydrogens=equivalize_hydrogens,
            equivalize_other_atoms=equivalize_other_atoms,
        )

    state = setup_from_gdma_record(gdma_record, atom_type_labels)
    result = optimize_constrained(state, verbose=verbose)
    return result["qstore"]


def fit_transferable_charges(
    gdma_records: list[MoleculeGDMARecord],
    atom_type_labels: list[str],
    molecule_charge: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Fit transferable charges across multiple molecules.

    Atoms with the same type label across different molecules are constrained
    to have identical total charges, enabling transferable force field parameters.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord]
        GDMA records for each molecule to fit simultaneously
    atom_type_labels : list[str]
        Combined atom type labels for all atoms across all molecules.
        Length must equal sum of atoms in all molecules.
        Use the same label for atoms that should share charges.
    molecule_charge : float
        Total charge constraint (sum of all molecules' charges)
    verbose : bool
        Print optimization progress

    Returns
    -------
    dict with keys:
        'qstore': np.ndarray - Fitted charges for all atoms (concatenated)
        'charges_by_molecule': list[np.ndarray] - Charges split by molecule
        'atom_counts': list[int] - Number of atoms per molecule
        'unique_charges': dict[str, float] - Charge for each unique atom type
    """
    from openff.toolkit import Molecule

    # Get atom counts for each molecule
    atom_counts = []
    for gdma_record in gdma_records:
        molecule = Molecule.from_mapped_smiles(
            gdma_record.tagged_smiles, allow_undefined_stereo=True
        )
        atom_counts.append(molecule.n_atoms)

    # Setup combined state
    state = setup_from_multiple_gdma_records(gdma_records, atom_type_labels)
    state.molecule_charge = molecule_charge

    if verbose:
        print(f"Fitting {len(gdma_records)} molecules with {sum(atom_counts)} total atoms")
        print(f"Unique atom types: {len(set(atom_type_labels))}")

    # Optimize
    result = optimize_constrained(state, verbose=verbose)

    # Split charges by molecule
    charges_by_molecule = []
    offset = 0
    for count in atom_counts:
        charges_by_molecule.append(result["qstore"][offset:offset + count])
        offset += count

    # Extract unique charges per atom type
    unique_charges = {}
    for label, charge in zip(atom_type_labels, result["qstore"]):
        if label not in unique_charges:
            unique_charges[label] = charge

    return {
        "qstore": result["qstore"],
        "charges_by_molecule": charges_by_molecule,
        "atom_counts": atom_counts,
        "unique_charges": unique_charges,
        "objective": result["objective"],
        "success": result["success"],
    }


# =============================================================================
# Manual Atom Typing for Ionic Liquids
# =============================================================================


def generate_imidazolium_atom_types(molecule) -> list[str]:
    """Generate manual atom type labels for imidazolium cations (EMIM, BMIM, C6MIM)."""
    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    rdmol = molecule.to_rdkit()
    ring_atoms = set()
    for ring in rdmol.GetRingInfo().AtomRings():
        if len(ring) == 5:
            ring_atoms.update(ring)

    ring_nitrogens = [i for i in ring_atoms if molecule.atoms[i].atomic_number == 7]
    ring_carbons = [i for i in ring_atoms if molecule.atoms[i].atomic_number == 6]

    c2_atom, c45_atoms = None, []
    for c in ring_carbons:
        n_neighbors = sum(1 for b in molecule.atoms[c].bonds
                        if (b.atom1_index if b.atom2_index == c else b.atom2_index) in ring_nitrogens)
        if n_neighbors == 2:
            c2_atom = c
        else:
            c45_atoms.append(c)

    for n in ring_nitrogens:
        labels[n] = "N_ring"
    if c2_atom is not None:
        labels[c2_atom] = "C_ring_2"
    for c in c45_atoms:
        labels[c] = "C_ring_45"

    alkyl_carbons = [i for i in range(n_atoms)
                    if molecule.atoms[i].atomic_number == 6 and i not in ring_atoms]

    for c in alkyl_carbons:
        h_count = sum(1 for b in molecule.atoms[c].bonds
                     if molecule.atoms[b.atom1_index if b.atom2_index == c else b.atom2_index].atomic_number == 1)
        if h_count == 3:
            labels[c] = "C_CH3"
        elif h_count == 2:
            dist = _distance_from_ring(molecule, c, ring_atoms)
            labels[c] = f"C_CH2_{dist}"

    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            parent = next((b.atom1_index if b.atom2_index == i else b.atom2_index
                          for b in molecule.atoms[i].bonds), None)
            if parent is not None:
                parent_label = labels[parent]
                if parent_label == "C_ring_2":
                    labels[i] = "H_ring_2"
                elif parent_label == "C_ring_45":
                    labels[i] = "H_ring_45"
                elif parent_label == "C_CH3":
                    labels[i] = "H_CH3"
                elif parent_label.startswith("C_CH2"):
                    labels[i] = "H_CH2"

    return labels


def generate_quaternary_ammonium_atom_types(molecule) -> list[str]:
    """Generate manual atom type labels for quaternary ammonium/phosphonium (N4444, P4444)."""
    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    central_atom = next((i for i in range(n_atoms)
                        if molecule.atoms[i].atomic_number in [7, 15]
                        and int(molecule.atoms[i].formal_charge.magnitude) == 1), None)

    if central_atom is None:
        raise ValueError("Could not find central N+ or P+")

    element = "N" if molecule.atoms[central_atom].atomic_number == 7 else "P"
    labels[central_atom] = f"{element}_center"

    carbon_distances = {}
    visited = {central_atom}
    queue = [(central_atom, 0)]

    while queue:
        current, dist = queue.pop(0)
        for b in molecule.atoms[current].bonds:
            neighbor = b.atom1_index if b.atom2_index == current else b.atom2_index
            if neighbor not in visited:
                visited.add(neighbor)
                if molecule.atoms[neighbor].atomic_number == 6:
                    carbon_distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                elif molecule.atoms[neighbor].atomic_number != 1:
                    queue.append((neighbor, dist + 1))

    distance_names = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
    for c, dist in carbon_distances.items():
        h_count = sum(1 for b in molecule.atoms[c].bonds
                     if molecule.atoms[b.atom1_index if b.atom2_index == c else b.atom2_index].atomic_number == 1)
        if h_count == 3:
            labels[c] = "C_CH3"
        else:
            labels[c] = f"C_{distance_names.get(dist, f'd{dist}')}"

    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            parent = next((b.atom1_index if b.atom2_index == i else b.atom2_index
                          for b in molecule.atoms[i].bonds), None)
            if parent is not None and labels[parent]:
                labels[i] = f"H_{labels[parent].replace('C_', '')}"

    return labels


def _distance_from_ring(molecule, atom_idx: int, ring_atoms: set) -> int:
    """BFS to find shortest distance from atom to ring."""
    from collections import deque
    visited = {atom_idx}
    queue = deque([(atom_idx, 0)])

    while queue:
        current, dist = queue.popleft()
        for b in molecule.atoms[current].bonds:
            neighbor = b.atom1_index if b.atom2_index == current else b.atom2_index
            if neighbor in ring_atoms:
                return dist + 1
            if neighbor not in visited and molecule.atoms[neighbor].atomic_number == 6:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 99


def get_manual_atom_types(molecule, mol_key: str) -> list[str] | None:
    """Get manual atom type labels for specific molecule types."""
    if mol_key in ["emim", "bmim", "c6mim"]:
        return generate_imidazolium_atom_types(molecule)
    elif mol_key in ["n4444", "p4444"]:
        return generate_quaternary_ammonium_atom_types(molecule)
    return None


# =============================================================================
# Test Molecules and Main
# =============================================================================

TEST_MOLECULES = {
    "ethanol": ("Ethanol", "CCO", 0),
    "emim": ("1-ethyl-3-methylimidazolium (EMIM)", "CCn1cc[n+](C)c1", 1),
    "bmim": ("1-butyl-3-methylimidazolium (BMIM)", "CCCCn1cc[n+](C)c1", 1),
    "c6mim": ("1-hexyl-3-methylimidazolium (C6MIM)", "CCCCCCn1cc[n+](C)c1", 1),
    "n4444": ("Tetrabutylammonium (N4444)", "CCCC[N+](CCCC)(CCCC)CCCC", 1),
    "p4444": ("Tetrabutylphosphonium (P4444)", "CCCC[P+](CCCC)(CCCC)CCCC", 1),
}


def test_molecule(
    name: str,
    smiles: str,
    expected_charge: int = 0,
    verbose: bool = True,
    verify_grad: bool = True,
    mol_key: str | None = None,
) -> dict:
    """Test constrained MPFIT on a single molecule."""
    import time
    from openff.toolkit import Molecule
    from openff.recharge.utilities.molecule import extract_conformers
    from openff.units.elements import SYMBOLS
    from openff_pympfit.gdma import GDMASettings
    from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
    from openff_pympfit.gdma.storage import MoleculeGDMARecord

    timings = {}
    t_total_start = time.time()

    print("\n" + "=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)

    t_start = time.time()
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    timings["molecule_setup"] = time.time() - t_start

    if verbose:
        print(f"\nSMILES: {smiles}")
        print(f"Atoms: {molecule.n_atoms}, Charge: {molecule.total_charge}")
        print(f"  [Setup: {timings['molecule_setup']:.2f}s]")

    if verbose:
        print("\n--- Psi4 GDMA ---")
    gdma_settings = GDMASettings()

    t_start = time.time()
    conformer, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, gdma_settings, minimize=True)
    gdma_record = MoleculeGDMARecord.from_molecule(molecule, conformer, multipoles, gdma_settings)
    timings["psi4_gdma"] = time.time() - t_start

    if verbose:
        print(f"  [Psi4: {timings['psi4_gdma']:.2f}s ({timings['psi4_gdma']/60:.2f} min)]")

    t_start = time.time()
    labels = get_manual_atom_types(molecule, mol_key) if mol_key else None
    if labels is None:
        labels = generate_atom_type_labels_from_symmetry(molecule)
        label_source = "auto"
    else:
        label_source = "manual"
    timings["atom_typing"] = time.time() - t_start

    if verbose:
        print(f"\n--- Atom Types ({label_source}) ---")
        print(f"Labels: {labels}")

    equiv_classes = {}
    for i, label in enumerate(labels):
        equiv_classes.setdefault(label, []).append(i)

    if verbose:
        print("\nEquivalence classes:")
        for label, indices in equiv_classes.items():
            atoms_str = ", ".join(f"{i}({SYMBOLS[molecule.atoms[i].atomic_number]})" for i in indices)
            print(f"  {label}: [{atoms_str}]")

    t_start = time.time()
    state = setup_from_gdma_record(gdma_record, labels)
    state.molecule_charge = float(expected_charge)
    n_full = int(np.sum(state.quse))
    n_reduced = count_parameters(state)
    timings["mpfit_setup"] = time.time() - t_start

    if verbose:
        print(f"\n--- Setup ---")
        print(f"Params: {n_reduced}/{n_full} (saved {n_full - n_reduced})")

    if verify_grad:
        t_start = time.time()
        p0_test = np.random.randn(n_reduced) * 0.01
        _, _, max_error = verify_gradient(p0_test, state)
        timings["gradient_verify"] = time.time() - t_start
        if verbose:
            print(f"\n--- Gradient Check ---")
            print(f"Max error: {max_error:.2e} [{'PASS' if max_error < 1e-4 else 'FAIL'}]")
            print(f"  [{timings['gradient_verify']:.2f}s]")

    if verbose:
        print("\n--- Optimization ---")
    t_start = time.time()
    result = optimize_constrained(state, verbose=verbose)
    timings["optimization"] = time.time() - t_start

    if verbose:
        print(f"  [{timings['optimization']:.2f}s ({timings['optimization']/60:.2f} min)]")

    if verbose:
        print("\n--- Final Charges ---")
        for i, (label, q) in enumerate(zip(labels, result["qstore"])):
            print(f"  {i:2d} ({SYMBOLS[molecule.atoms[i].atomic_number]:2s}, {label}): {q:+.6f}")

    all_satisfied = True
    constraint_results = {}
    for label, indices in equiv_classes.items():
        if len(indices) > 1:
            charges = [result["qstore"][i] for i in indices]
            max_diff = max(charges) - min(charges)
            satisfied = max_diff < 1e-10
            if not satisfied:
                all_satisfied = False
            constraint_results[label] = {"max_diff": max_diff, "satisfied": satisfied}

    if verbose:
        print("\n--- Constraints ---")
        for label, info in constraint_results.items():
            print(f"  {label}: {info['max_diff']:.2e} [{'PASS' if info['satisfied'] else 'FAIL'}]")
        print(f"\nTotal charge: {np.sum(result['qstore']):.6f} (expected: {expected_charge})")

    timings["total"] = time.time() - t_total_start

    if verbose:
        print(f"\n--- Timing ---")
        print(f"  Psi4: {timings['psi4_gdma']:.1f}s, Opt: {timings['optimization']:.1f}s, Total: {timings['total']:.1f}s")

    return {
        "name": name, "n_atoms": molecule.n_atoms, "qstore": result["qstore"],
        "labels": labels, "all_satisfied": all_satisfied,
        "total_charge": np.sum(result["qstore"]), "timings": timings,
    }


def main():
    """Test constrained MPFIT implementation."""
    import sys

    print("=" * 70)
    print("Constrained MPFIT Implementation")
    print("=" * 70)

    if len(sys.argv) > 1:
        molecule_key = sys.argv[1].lower()
        if molecule_key == "all":
            molecules_to_test = list(TEST_MOLECULES.keys())
        elif molecule_key in TEST_MOLECULES:
            molecules_to_test = [molecule_key]
        else:
            print(f"Unknown: {molecule_key}. Available: {', '.join(TEST_MOLECULES.keys())}, all")
            sys.exit(1)
    else:
        molecules_to_test = ["ethanol"]

    print(f"Testing: {molecules_to_test}")

    results = {}
    for mol_key in molecules_to_test:
        name, smiles, expected_charge = TEST_MOLECULES[mol_key]
        try:
            results[mol_key] = test_molecule(name, smiles, expected_charge, mol_key=mol_key)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    if len(molecules_to_test) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for mol_key, result in results.items():
            status = "PASS" if result.get("all_satisfied") else "FAIL"
            print(f"  {mol_key}: {result['n_atoms']} atoms, q={result['total_charge']:.4f} [{status}]")


if __name__ == "__main__":
    main()
