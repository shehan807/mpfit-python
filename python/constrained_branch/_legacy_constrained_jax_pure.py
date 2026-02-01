"""
Constrained MPFIT Implementation

Fits partial charges to reproduce GDMA multipoles with atom-type equivalence constraints.
Atoms with the same type label are constrained to have equal total charges.
"""

from __future__ import annotations

from functools import partial
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
    atom_counts: list[int] = field(default_factory=list)


# --- Setup ---

def setup_from_gdma_records(
    gdma_records: list[MoleculeGDMARecord] | MoleculeGDMARecord,
    atom_type_labels: list[str],
) -> ConstrainedMPFITState:
    """Initialize state from one or more GDMA records.

    For multiple molecules, coordinates and multipoles are stacked into a
    combined system.  Atoms with matching labels (even across different
    molecules) are constrained to share the same charge.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord] or MoleculeGDMARecord
        One or more GDMA records.
    atom_type_labels : list[str]
        Atom type labels for all atoms (concatenated across molecules).
        Length must equal the total number of atoms.
    """
    from openff.toolkit import Molecule
    from openff.units import unit
    from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical

    if not isinstance(gdma_records, list):
        gdma_records = [gdma_records]

    all_xyz = []
    all_multipoles = []
    all_rvdw = []
    all_lmax = []
    atom_counts = []
    total_atoms = 0

    for gdma_record in gdma_records:
        molecule = Molecule.from_mapped_smiles(
            gdma_record.tagged_smiles, allow_undefined_stereo=True
        )
        n_atoms = molecule.n_atoms
        total_atoms += n_atoms
        atom_counts.append(n_atoms)
        gdma_settings = gdma_record.gdma_settings

        conformer_bohr = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
        all_xyz.append(conformer_bohr)

        multipoles = _convert_flat_to_hierarchical(
            gdma_record.multipoles, n_atoms, gdma_settings.limit
        )
        all_multipoles.append(multipoles)

        all_rvdw.append(np.full(n_atoms, gdma_settings.mpfit_atom_radius))
        all_lmax.append(np.full(n_atoms, gdma_settings.limit, dtype=float))

    if len(atom_type_labels) != total_atoms:
        raise ValueError(
            f"atom_type_labels has {len(atom_type_labels)} entries, "
            f"but total atoms across all molecules is {total_atoms}"
        )

    gdma_settings = gdma_records[0].gdma_settings

    state = ConstrainedMPFITState()
    state.r1 = gdma_settings.mpfit_inner_radius
    state.r2 = gdma_settings.mpfit_outer_radius
    state.atomtype = atom_type_labels

    state.xyzcharge = np.vstack(all_xyz)
    state.xyzmult = np.vstack(all_xyz)
    state.multipoles = np.vstack(all_multipoles)
    state.rvdw = np.concatenate(all_rvdw)
    state.lmax = np.concatenate(all_lmax)

    state.atom_counts = atom_counts
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
    from scipy.spatial.distance import cdist
    return (cdist(xyzmult, xyzcharge) < rvdw[:, None]).astype(int)


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


# --- Core Algorithm ---

def build_constraint_matrix(state: ConstrainedMPFITState) -> np.ndarray:
    """Build linear constraint matrix M such that allcharge = (M @ p0).reshape(n_atoms, n_sites).T.

    Encodes atom-type equivalence constraints as a linear map from reduced
    parameters p0 to the full allcharge matrix.  Replaces the imperative
    expandcharge function with a single matrix multiply.

    Returns
    -------
    M : ndarray, shape (n_atoms * n_sites, n_params)
    """
    atomtype = state.atomtype
    quse = state.quse
    n_sites = quse.shape[0]
    n_atoms = len(atomtype)
    n_params = count_parameters(state)

    M = np.zeros((n_atoms * n_sites, n_params))
    count = 0

    for i in range(n_atoms):
        if i == 0:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    M[i * n_sites + j, count] = 1.0
                    count += 1
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)

            if twin is not None:
                count1 = int(np.sum(quse[:, i]))
                count2 = 1
                for j in range(n_sites):
                    if quse[j, i] == 1 and count2 < count1:
                        M[i * n_sites + j, count] = 1.0
                        count += 1
                        count2 += 1
                    elif quse[j, i] == 1 and count2 == count1:
                        # Last active site: qstore[twin] - charge_sum
                        # = sum of twin's active rows - sum of i's earlier active rows
                        twin_row_sum = np.zeros(n_params)
                        for k in range(n_sites):
                            if quse[k, twin] == 1:
                                twin_row_sum += M[twin * n_sites + k, :]
                        earlier_row_sum = np.zeros(n_params)
                        for k in range(n_sites):
                            if quse[k, i] == 1 and k < j:
                                earlier_row_sum += M[i * n_sites + k, :]
                        M[i * n_sites + j, :] = twin_row_sum - earlier_row_sum
            else:
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        M[i * n_sites + j, count] = 1.0
                        count += 1

    return M


def build_constraint_matrices(state: ConstrainedMPFITState) -> list[np.ndarray]:
    """Build per-molecule constraint matrices for block-diagonal evaluation.

    Returns a list of M_i arrays, one per molecule. Each M_i has shape
    (n_atoms_i * n_atoms_i, n_params) and maps the global parameter vector
    p0 to that molecule's charge matrix:
        allcharge_i = (M_i @ p0).reshape(n_atoms_i, n_atoms_i).T

    The global parameter vector is shared across all molecules, with atoms
    of the same type referencing the same parameter columns.
    """
    atomtype = state.atomtype
    quse = state.quse
    n_sites_global = quse.shape[0]
    n_atoms_global = len(atomtype)
    n_params = count_parameters(state)
    atom_counts = state.atom_counts

    # Compute molecule offsets
    offsets = []
    s = 0
    for c in atom_counts:
        offsets.append(s)
        s += c

    # Map each global atom index to (molecule_index, local_atom_index)
    atom_to_mol = {}
    for mol_idx, (offset, n_atoms_mol) in enumerate(zip(offsets, atom_counts)):
        for local_i in range(n_atoms_mol):
            atom_to_mol[offset + local_i] = (mol_idx, local_i)

    # Allocate per-molecule M blocks
    M_blocks = [np.zeros((nc * nc, n_params)) for nc in atom_counts]

    # Build using same global iteration as build_constraint_matrix,
    # but store rows in per-molecule blocks
    count = 0

    for i in range(n_atoms_global):
        mol_idx, local_i = atom_to_mol[i]
        mol_offset = offsets[mol_idx]
        n_mol = atom_counts[mol_idx]

        # Only iterate over this molecule's sites (block-diagonal quse)
        site_start = mol_offset
        site_end = mol_offset + n_mol

        if i == 0:
            for j in range(site_start, site_end):
                local_j = j - site_start
                if quse[j, i] == 1:
                    M_blocks[mol_idx][local_i * n_mol + local_j, count] = 1.0
                    count += 1
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)

            if twin is not None:
                count1 = int(np.sum(quse[site_start:site_end, i]))
                count2 = 1
                for j in range(site_start, site_end):
                    local_j = j - site_start
                    if quse[j, i] == 1 and count2 < count1:
                        M_blocks[mol_idx][local_i * n_mol + local_j, count] = 1.0
                        count += 1
                        count2 += 1
                    elif quse[j, i] == 1 and count2 == count1:
                        # Twin row sum — twin may be in a different molecule
                        twin_mol_idx, twin_local_i = atom_to_mol[twin]
                        twin_offset = offsets[twin_mol_idx]
                        twin_n_mol = atom_counts[twin_mol_idx]
                        twin_site_start = twin_offset
                        twin_site_end = twin_offset + twin_n_mol

                        twin_row_sum = np.zeros(n_params)
                        for k in range(twin_site_start, twin_site_end):
                            twin_local_k = k - twin_site_start
                            if quse[k, twin] == 1:
                                twin_row_sum += M_blocks[twin_mol_idx][twin_local_i * twin_n_mol + twin_local_k, :]

                        earlier_row_sum = np.zeros(n_params)
                        for k in range(site_start, site_end):
                            local_k = k - site_start
                            if quse[k, i] == 1 and k < j:
                                earlier_row_sum += M_blocks[mol_idx][local_i * n_mol + local_k, :]

                        M_blocks[mol_idx][local_i * n_mol + local_j, :] = twin_row_sum - earlier_row_sum
            else:
                for j in range(site_start, site_end):
                    local_j = j - site_start
                    if quse[j, i] == 1:
                        M_blocks[mol_idx][local_i * n_mol + local_j, count] = 1.0
                        count += 1

    return M_blocks


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


def kaisq_pure(p0, M, multipoles, xyzmult, xyzcharge, rvdw, lmax,
               maxl, r1, r2, molecule_charge, conchg, xp=np):
    """Pure objective function: no state mutation, all inputs are arrays.

    Vectorized over sites — the only Python loop is over (l, m, cs) components
    (25 iterations for maxl=4), which JAX traces into a fixed-size graph.

    Parameters
    ----------
    p0 : array, shape (n_params,)
        Reduced parameter vector.
    M : array, shape (n_atoms * n_sites, n_params)
        Constraint matrix from build_constraint_matrix.
    multipoles, xyzmult, xyzcharge : arrays
        Geometry and multipole data (from state).
    rvdw : list[float] or array
        Van der Waals radii per site.
    lmax : list[int] or array
        Max multipole rank per site.
    maxl, r1, r2, molecule_charge, conchg : scalars
        Settings (from state).
    xp : module
        Array backend (np or jnp).
    """
    n_sites = xyzmult.shape[0]
    n_atoms = len(rvdw)

    allcharge = (xp.dot(M, p0)).reshape(n_atoms, n_sites).T  # (n_sites, n_atoms)
    qstore = xp.sum(allcharge, axis=0)

    # Displacements: (n_sites, n_atoms)
    dx = xyzcharge[None, :, 0] - xyzmult[:, 0, None]
    dy = xyzcharge[None, :, 1] - xyzmult[:, 1, None]
    dz = xyzcharge[None, :, 2] - xyzmult[:, 2, None]

    # W weights: (n_sites, maxl+1)
    rvdw_arr = xp.asarray(rvdw, dtype=xp.float64) if not isinstance(rvdw, list) else np.array(rvdw)
    rmax = rvdw_arr + r2   # (n_sites,)
    rminn = rvdw_arr + r1  # (n_sites,)

    W = np.zeros((n_sites, maxl + 1))
    for i in range(maxl + 1):
        W[:, i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

    # Build (l, m, cs) component list
    components = []
    for l in range(maxl + 1):
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                components.append((l, m, cs))

    sumkai = 0.0
    for l, m, cs in components:
        angular = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
        weight = angular * W[:, l]  # (n_sites,)

        rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz, xp=xp)  # (n_sites, n_atoms)
        sum1 = xp.sum(allcharge * rsh_vals, axis=1)  # (n_sites,)
        residuals = multipoles[:, l, m, cs] - sum1    # (n_sites,)
        sumkai += xp.sum(weight * residuals ** 2)

    sumchg = xp.sum(qstore)
    sumcon = conchg * (sumchg - molecule_charge) ** 2
    return sumkai + sumcon


def dkaisq_pure(p0, M, multipoles, xyzmult, xyzcharge, rvdw, lmax,
                maxl, r1, r2, molecule_charge, conchg):
    """Analytical gradient of kaisq_pure w.r.t. p0.

    Vectorized over sites. Uses M.T @ d_allcharge to map the full-space
    gradient back to reduced parameters, replacing createdkaisq.
    """
    n_sites = xyzmult.shape[0]
    n_atoms = len(rvdw)

    allcharge = (M @ p0).reshape(n_atoms, n_sites).T  # (n_sites, n_atoms)
    qstore = allcharge.sum(axis=0)

    # Displacements: (n_sites, n_atoms)
    dx = xyzcharge[None, :, 0] - xyzmult[:, 0, None]
    dy = xyzcharge[None, :, 1] - xyzmult[:, 1, None]
    dz = xyzcharge[None, :, 2] - xyzmult[:, 2, None]

    # W weights: (n_sites, maxl+1)
    rvdw_arr = np.array(rvdw) if isinstance(rvdw, list) else rvdw
    rmax = rvdw_arr + r2
    rminn = rvdw_arr + r1
    W = np.zeros((n_sites, maxl + 1))
    for i in range(maxl + 1):
        W[:, i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

    components = []
    for l in range(maxl + 1):
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                components.append((l, m, cs))

    d_allcharge = np.zeros((n_sites, n_atoms))
    for l, m, cs in components:
        angular = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
        weight = angular * W[:, l]  # (n_sites,)

        rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)  # (n_sites, n_atoms)
        sum1 = np.sum(allcharge * rsh_vals, axis=1)  # (n_sites,)
        coeff = 2.0 * weight * (multipoles[:, l, m, cs] - sum1)  # (n_sites,)
        d_allcharge -= coeff[:, None] * rsh_vals

    # Chain rule: d_allcharge (n_sites, n_atoms) → flat in (n_atoms, n_sites) order → M.T @
    d_flat = d_allcharge.T.flatten()

    # Add charge constraint gradient
    sumchg = np.sum(qstore)
    d_flat += conchg * 2.0 * (sumchg - molecule_charge)

    # Map back to reduced parameters
    return M.T @ d_flat


def kaisq_block(p0, M_blocks, mol_data, maxl, r1, r2, molecule_charge, conchg, xp=np):
    """Block-diagonal objective: sum of per-molecule contributions.

    Parameters
    ----------
    p0 : array, shape (n_params,)
        Global reduced parameter vector (shared across all molecules).
    M_blocks : list of arrays
        Per-molecule constraint matrices. M_blocks[i] has shape
        (n_atoms_i * n_atoms_i, n_params).
    mol_data : list of tuples
        Per-molecule data: (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i).
    maxl, r1, r2, molecule_charge, conchg : scalars
    xp : module (np or jnp)
    """
    # Build (l, m, cs) component list
    components = []
    for l in range(maxl + 1):
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                components.append((l, m, cs))

    total_kai = 0.0
    total_charge = 0.0

    for M_i, (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) in zip(M_blocks, mol_data):
        n_atoms_i = xyzcharge_i.shape[0]
        n_sites_i = xyzmult_i.shape[0]  # == n_atoms_i

        allcharge_i = (xp.dot(M_i, p0)).reshape(n_atoms_i, n_sites_i).T
        qstore_i = xp.sum(allcharge_i, axis=0)
        total_charge = total_charge + xp.sum(qstore_i)

        dx = xyzcharge_i[None, :, 0] - xyzmult_i[:, 0, None]
        dy = xyzcharge_i[None, :, 1] - xyzmult_i[:, 1, None]
        dz = xyzcharge_i[None, :, 2] - xyzmult_i[:, 2, None]

        rvdw_arr = xp.asarray(rvdw_i, dtype=xp.float64) if not isinstance(rvdw_i, list) else np.array(rvdw_i)
        rmax = rvdw_arr + r2
        rminn = rvdw_arr + r1

        W = np.zeros((n_sites_i, maxl + 1))
        for ii in range(maxl + 1):
            W[:, ii] = (1.0 / (1.0 - 2.0 * ii)) * (rmax ** (1 - 2 * ii) - rminn ** (1 - 2 * ii))

        for l, m, cs in components:
            angular = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight = angular * W[:, l]
            rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz, xp=xp)
            sum1 = xp.sum(allcharge_i * rsh_vals, axis=1)
            residuals = multipoles_i[:, l, m, cs] - sum1
            total_kai = total_kai + xp.sum(weight * residuals ** 2)

    total_kai = total_kai + conchg * (total_charge - molecule_charge) ** 2
    return total_kai


def dkaisq_block(p0, M_blocks, mol_data, maxl, r1, r2, molecule_charge, conchg):
    """Analytical gradient of kaisq_block w.r.t. p0."""
    components = []
    for l in range(maxl + 1):
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                components.append((l, m, cs))

    n_params = M_blocks[0].shape[1]
    grad = np.zeros(n_params)
    total_charge = 0.0

    for M_i, (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) in zip(M_blocks, mol_data):
        n_atoms_i = xyzcharge_i.shape[0]
        n_sites_i = xyzmult_i.shape[0]

        allcharge_i = (M_i @ p0).reshape(n_atoms_i, n_sites_i).T
        qstore_i = allcharge_i.sum(axis=0)
        total_charge += np.sum(qstore_i)

        dx = xyzcharge_i[None, :, 0] - xyzmult_i[:, 0, None]
        dy = xyzcharge_i[None, :, 1] - xyzmult_i[:, 1, None]
        dz = xyzcharge_i[None, :, 2] - xyzmult_i[:, 2, None]

        rvdw_arr = np.array(rvdw_i) if isinstance(rvdw_i, list) else rvdw_i
        rmax = rvdw_arr + r2
        rminn = rvdw_arr + r1
        W = np.zeros((n_sites_i, maxl + 1))
        for ii in range(maxl + 1):
            W[:, ii] = (1.0 / (1.0 - 2.0 * ii)) * (rmax ** (1 - 2 * ii) - rminn ** (1 - 2 * ii))

        d_allcharge_i = np.zeros((n_sites_i, n_atoms_i))
        for l, m, cs in components:
            angular = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight = angular * W[:, l]
            rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
            sum1 = np.sum(allcharge_i * rsh_vals, axis=1)
            coeff = 2.0 * weight * (multipoles_i[:, l, m, cs] - sum1)
            d_allcharge_i -= coeff[:, None] * rsh_vals

        d_flat_i = d_allcharge_i.T.flatten()
        grad += M_i.T @ d_flat_i

    # Charge constraint gradient (applied globally)
    charge_grad = conchg * 2.0 * (total_charge - molecule_charge)
    for M_i in M_blocks:
        # Each element of allcharge contributes 1.0 to total_charge via qstore sum
        grad += charge_grad * M_i.T @ np.ones(M_i.shape[0])

    return grad


def kaisq(p0: np.ndarray, state: ConstrainedMPFITState, xp=np) -> float:
    """Objective function: sum of squared multipole errors."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    maxl = state.maxl
    sumkai = 0.0

    for s in range(n_sites):
        q0 = xp.asarray(state.allcharge[s, :])
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        dx = xp.asarray(state.xyzcharge[:, 0] - state.xyzmult[s, 0])
        dy = xp.asarray(state.xyzcharge[:, 1] - state.xyzmult[s, 1])
        dz = xp.asarray(state.xyzcharge[:, 2] - state.xyzmult[s, 2])

        site_sum = 0.0
        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz, xp=xp)
                    sum1 = xp.dot(q0, rsh_vals)
                    site_sum += weight * (state.multipoles[s, l, m, cs] - float(sum1)) ** 2

        sumkai += site_sum

    sumchg = np.sum(state.qstore)
    sumcon = state.conchg * (sumchg - state.molecule_charge) ** 2
    return sumkai + sumcon


def dkaisq(p0: np.ndarray, state: ConstrainedMPFITState, xp=np) -> np.ndarray:
    """Gradient of kaisq with respect to reduced parameters."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    n_atoms = n_sites
    maxl = state.maxl
    dparam = np.zeros((n_sites, n_atoms))

    for s in range(n_sites):
        q0 = xp.asarray(state.allcharge[s, :])
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        dx = xp.asarray(state.xyzcharge[:, 0] - state.xyzmult[s, 0])
        dy = xp.asarray(state.xyzcharge[:, 1] - state.xyzmult[s, 1])
        dz = xp.asarray(state.xyzcharge[:, 2] - state.xyzmult[s, 2])

        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz, xp=xp)
                    sum1 = xp.dot(q0, rsh_vals)
                    coeff = 2.0 * weight * (state.multipoles[s, l, m, cs] - float(sum1))
                    dparam[s, :] -= coeff * np.asarray(rsh_vals)

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


# --- Optimization ---

def optimize_constrained(
    state: ConstrainedMPFITState,
    p0_init: np.ndarray | None = None,
    xp=np,
) -> dict:
    """Run constrained MPFIT optimization."""
    n_params = count_parameters(state)
    if p0_init is None:
        p0_init = np.zeros(n_params)

    result = minimize(
        fun=lambda p: kaisq(p, state, xp=xp),
        x0=p0_init,
        jac=lambda p: dkaisq(p, state, xp=xp),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10},
    )

    expandcharge(result.x, state)

    return {
        "qstore": state.qstore.copy(),
        "allcharge": state.allcharge.copy(),
        "objective": result.fun,
        "success": result.success,
        "scipy_result": result,
    }


# --- Solver Classes ---

def _slice_mol_data(state: ConstrainedMPFITState) -> list[tuple]:
    """Slice per-molecule data from the global state arrays.

    Returns list of (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) tuples.
    """
    mol_data = []
    offset = 0
    for n_atoms_i in state.atom_counts:
        mol_data.append((
            state.multipoles[offset:offset + n_atoms_i],
            state.xyzmult[offset:offset + n_atoms_i],
            state.xyzcharge[offset:offset + n_atoms_i],
            state.rvdw[offset:offset + n_atoms_i],
        ))
        offset += n_atoms_i
    return mol_data


class SciPySolver:
    """Numpy solver using block-diagonal constraint matrices."""

    def __init__(self, state: ConstrainedMPFITState):
        self.state = state
        self.M_blocks = build_constraint_matrices(state)
        self.mol_data = _slice_mol_data(state)
        self.n_params = count_parameters(state)

    def objective(self, p0):
        s = self.state
        return kaisq_block(p0, self.M_blocks, self.mol_data,
                           s.maxl, s.r1, s.r2, s.molecule_charge, s.conchg)

    def gradient(self, p0):
        s = self.state
        return dkaisq_block(p0, self.M_blocks, self.mol_data,
                            s.maxl, s.r1, s.r2, s.molecule_charge, s.conchg)

    def optimize(self, p0_init=None):
        if p0_init is None:
            p0_init = np.zeros(self.n_params)

        result = minimize(
            fun=self.objective,
            x0=p0_init,
            jac=self.gradient,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10},
        )

        # Reconstruct per-atom charges from block-diagonal M
        qstore_parts = []
        for M_i, (_, _, xyzcharge_i, _) in zip(self.M_blocks, self.mol_data):
            n_atoms_i = xyzcharge_i.shape[0]
            allcharge_i = (M_i @ result.x).reshape(n_atoms_i, n_atoms_i).T
            qstore_parts.append(allcharge_i.sum(axis=0))
        qstore = np.concatenate(qstore_parts)

        return {
            "qstore": qstore,
            "objective": result.fun,
            "success": result.success,
            "scipy_result": result,
        }


class DirectSolver:
    """Direct linear solve — exploits the quadratic structure of the objective.

    The MPFIT objective is quadratic in p0:
        f(p0) = p0^T H p0 - 2 g^T p0 + const + conchg*(c^T p0 - Q)^2

    The exact global minimum is found by solving the linear system:
        (H + conchg * c c^T) p0 = g + conchg * Q * c

    No iterative optimization, no JAX, no JIT compilation needed.
    """

    def __init__(self, state: ConstrainedMPFITState):
        self.state = state
        self.M_blocks = build_constraint_matrices(state)
        self.mol_data = _slice_mol_data(state)
        self.n_params = count_parameters(state)

    def solve(self):
        state = self.state
        n_params = self.n_params
        maxl = state.maxl
        r1, r2 = state.r1, state.r2

        # Build Hessian H and linear term g
        H = np.zeros((n_params, n_params))
        g = np.zeros(n_params)

        for M_i, (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) in zip(
            self.M_blocks, self.mol_data
        ):
            n_atoms_i = xyzcharge_i.shape[0]
            n_sites_i = n_atoms_i  # sites == atoms

            # W weights per site
            rvdw_arr = np.asarray(rvdw_i, dtype=np.float64)
            rmax = rvdw_arr + r2
            rminn = rvdw_arr + r1
            W = np.zeros((n_sites_i, maxl + 1))
            for ii in range(maxl + 1):
                W[:, ii] = (1.0 / (1.0 - 2.0 * ii)) * (
                    rmax ** (1 - 2 * ii) - rminn ** (1 - 2 * ii)
                )

            for s in range(n_sites_i):
                dx = xyzcharge_i[:, 0] - xyzmult_i[s, 0]
                dy = xyzcharge_i[:, 1] - xyzmult_i[s, 1]
                dz = xyzcharge_i[:, 2] - xyzmult_i[s, 2]

                A_s = np.zeros((n_atoms_i, n_atoms_i))
                b_s = np.zeros(n_atoms_i)

                for l in range(maxl + 1):
                    angular = 4.0 * np.pi / (2.0 * l + 1.0)
                    w = angular * W[s, l]
                    for m in range(l + 1):
                        cs_range = [0] if m == 0 else [0, 1]
                        for cs in cs_range:
                            rsh = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                            A_s += w * np.outer(rsh, rsh)
                            b_s += w * multipoles_i[s, l, m, cs] * rsh

                # Extract M_s: rows of M_i corresponding to site s
                # M_i @ p0 reshapes as (n_atoms_i, n_sites_i), then .T
                # allcharge[s, a] = flat[a * n_sites_i + s]
                row_indices = [a * n_sites_i + s for a in range(n_atoms_i)]
                M_s = M_i[row_indices, :]  # (n_atoms_i, n_params)

                H += M_s.T @ A_s @ M_s
                g += M_s.T @ b_s

        # Charge constraint: conchg * (c^T p0 - Q)^2
        # where c^T p0 = sum of all charges = ones^T @ (M @ p0)
        c_total = np.zeros(n_params)
        for M_i in self.M_blocks:
            c_total += M_i.T @ np.ones(M_i.shape[0])

        H_aug = H + state.conchg * np.outer(c_total, c_total)
        g_aug = g + state.conchg * state.molecule_charge * c_total

        # H is typically very rank-deficient (n_params >> effective DOF).
        # Solve in the non-null subspace via eigendecomposition for speed.
        eigvals, eigvecs = np.linalg.eigh(H_aug)
        tol = max(H_aug.shape) * np.max(np.abs(eigvals)) * np.finfo(float).eps
        mask = eigvals > tol
        # Project g into eigenbasis, solve, project back
        g_proj = eigvecs[:, mask].T @ g_aug
        p_proj = g_proj / eigvals[mask]
        p_opt = eigvecs[:, mask] @ p_proj

        # Reconstruct per-atom charges
        qstore_parts = []
        for M_i, (_, _, xyzcharge_i, _) in zip(self.M_blocks, self.mol_data):
            n_atoms_i = xyzcharge_i.shape[0]
            allcharge_i = (M_i @ p_opt).reshape(n_atoms_i, n_atoms_i).T
            qstore_parts.append(allcharge_i.sum(axis=0))
        qstore = np.concatenate(qstore_parts)

        # Compute objective for reporting
        obj = kaisq_block(
            p_opt, self.M_blocks, self.mol_data,
            state.maxl, r1, r2, state.molecule_charge, state.conchg,
        )

        return {
            "qstore": qstore,
            "objective": obj,
            "success": True,
            "p_opt": p_opt,
        }


class JAXSolver:
    """JAX JIT-compiled solver using block-diagonal constraint matrices."""

    def __init__(self, state: ConstrainedMPFITState):
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        self.state = state
        self.n_params = count_parameters(state)
        self._jnp = jnp
        self._jax = jax

        # Use GPU if available, otherwise CPU
        try:
            gpu_devices = jax.devices("gpu")
            self._device = gpu_devices[0]
        except RuntimeError:
            self._device = jax.devices("cpu")[0]
        print(f"  JAXSolver using device: {self._device}")

        # Build block-diagonal M and per-molecule data
        M_blocks_np = build_constraint_matrices(state)
        mol_data_np = _slice_mol_data(state)

        # Convert to JAX arrays on device
        self._M_blocks = [
            jax.device_put(jnp.array(M_i, dtype=jnp.float64), self._device)
            for M_i in M_blocks_np
        ]
        self._mol_data = [
            (
                jax.device_put(jnp.array(mp, dtype=jnp.float64), self._device),
                jax.device_put(jnp.array(xm, dtype=jnp.float64), self._device),
                jax.device_put(jnp.array(xc, dtype=jnp.float64), self._device),
                [float(x) for x in rv],  # Python list for loop control
            )
            for mp, xm, xc, rv in mol_data_np
        ]

        # Scalars (static, not traced)
        maxl = state.maxl
        r1 = state.r1
        r2 = state.r2
        molecule_charge = state.molecule_charge
        conchg = state.conchg

        # Capture in closure for JIT
        M_blocks = self._M_blocks
        mol_data = self._mol_data

        @jax.jit
        def objective(p0):
            return kaisq_block(p0, M_blocks, mol_data,
                               maxl, r1, r2, molecule_charge, conchg, xp=jnp)

        self._objective = objective

        # Warmup JIT compilation
        p0_warm = jnp.zeros(self.n_params, dtype=jnp.float64)
        _ = self._objective(p0_warm)

    def optimize(self, p0_init=None):
        from jaxopt import LBFGS
        jnp = self._jnp

        if p0_init is None:
            p0_init = jnp.zeros(self.n_params, dtype=jnp.float64)
        else:
            p0_init = jnp.array(p0_init, dtype=jnp.float64)

        solver = LBFGS(fun=self._objective, maxiter=1000, tol=1e-12,
                       linesearch="zoom", maxls=30, history_size=10)
        res = solver.run(p0_init)

        p_opt = np.array(res.params)

        # Reconstruct per-atom charges from block-diagonal M
        qstore_parts = []
        M_blocks_np = build_constraint_matrices(self.state)
        mol_data_np = _slice_mol_data(self.state)
        for M_i, (_, _, xyzcharge_i, _) in zip(M_blocks_np, mol_data_np):
            n_atoms_i = xyzcharge_i.shape[0]
            allcharge_i = (M_i @ p_opt).reshape(n_atoms_i, n_atoms_i).T
            qstore_parts.append(allcharge_i.sum(axis=0))
        qstore = np.concatenate(qstore_parts)

        obj_val = float(self._objective(res.params))

        return {
            "qstore": qstore,
            "objective": obj_val,
            "success": True,
        }


# --- High-Level API ---

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
    gdma_records: list[MoleculeGDMARecord] | MoleculeGDMARecord,
    atom_type_labels: list[str],
    molecule_charge: float = 0.0,
    backend: str = "numpy",
) -> dict:
    """Fit constrained MPFIT charges for one or more molecules.

    Atoms with the same type label are constrained to share the same charge,
    both within a molecule and across multiple molecules.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord] or MoleculeGDMARecord
        One or more GDMA records.
    atom_type_labels : list[str]
        Atom type labels for all atoms (concatenated across molecules).
        Length must equal the total number of atoms.
    molecule_charge : float
        Total charge constraint.
    backend : str
        Array backend: "numpy" or "jax".

    Returns
    -------
    dict with keys:
        'qstore': np.ndarray - Fitted charges for all atoms
        'allcharge': np.ndarray - Per-site charge distributions
        'objective': float - Final objective function value
        'success': bool - Whether the optimizer converged
        'scipy_result': OptimizeResult - Full scipy result
    """
    if backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        xp = jnp
    else:
        xp = np

    state = setup_from_gdma_records(gdma_records, atom_type_labels)
    state.molecule_charge = molecule_charge
    return optimize_constrained(state, xp=xp)


# --- Test / Main ---

def test_molecule(
    name: str,
    smiles: str,
    labels: list[str],
    expected_charge: int = 0,
) -> dict:
    """Test constrained MPFIT on a single molecule.

    Parameters
    ----------
    name : str
        Display name for the molecule.
    smiles : str
        SMILES string.
    labels : list[str]
        Atom type labels (one per atom). Equivalent atoms share the same label.
    expected_charge : int
        Net molecular charge.
    """
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

    print(f"\nSMILES: {smiles}")
    print(f"Atoms: {molecule.n_atoms}, Charge: {molecule.total_charge}")
    print(f"  [Setup: {timings['molecule_setup']:.2f}s]")

    print("\n--- Psi4 GDMA ---")
    gdma_settings = GDMASettings()

    t_start = time.time()
    conformer, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, gdma_settings, minimize=True)
    gdma_record = MoleculeGDMARecord.from_molecule(molecule, conformer, multipoles, gdma_settings)
    timings["psi4_gdma"] = time.time() - t_start

    print(f"  [Psi4: {timings['psi4_gdma']:.2f}s ({timings['psi4_gdma']/60:.2f} min)]")

    print(f"\n--- Atom Types ---")
    print(f"Labels: {labels}")

    equiv_classes = {}
    for i, label in enumerate(labels):
        equiv_classes.setdefault(label, []).append(i)

    print("\nEquivalence classes:")
    for label, indices in equiv_classes.items():
        atoms_str = ", ".join(f"{i}({SYMBOLS[molecule.atoms[i].atomic_number]})" for i in indices)
        print(f"  {label}: [{atoms_str}]")

    t_start = time.time()
    state = setup_from_gdma_records(gdma_record, labels)
    state.molecule_charge = float(expected_charge)
    n_full = int(np.sum(state.quse))
    n_reduced = count_parameters(state)
    timings["mpfit_setup"] = time.time() - t_start

    print(f"\n--- Setup ---")
    print(f"Params: {n_reduced}/{n_full} (saved {n_full - n_reduced})")

    print("\n--- Optimization ---")
    t_start = time.time()
    result = optimize_constrained(state)
    timings["optimization"] = time.time() - t_start

    print(f"  [{timings['optimization']:.2f}s ({timings['optimization']/60:.2f} min)]")

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

    print("\n--- Constraints ---")
    for label, info in constraint_results.items():
        print(f"  {label}: {info['max_diff']:.2e} [{'PASS' if info['satisfied'] else 'FAIL'}]")
    print(f"\nTotal charge: {np.sum(result['qstore']):.6f} (expected: {expected_charge})")

    timings["total"] = time.time() - t_total_start

    print(f"\n--- Timing ---")
    print(f"  Psi4: {timings['psi4_gdma']:.1f}s, Opt: {timings['optimization']:.1f}s, Total: {timings['total']:.1f}s")

    return {
        "name": name, "n_atoms": molecule.n_atoms, "qstore": result["qstore"],
        "labels": labels, "all_satisfied": all_satisfied,
        "total_charge": np.sum(result["qstore"]), "timings": timings,
    }


def main():
    """Test constrained MPFIT on ethanol."""
    from openff.toolkit import Molecule

    print("=" * 70)
    print("Constrained MPFIT — Ethanol Test")
    print("=" * 70)

    molecule = Molecule.from_smiles("CCO")
    labels = generate_atom_type_labels_from_symmetry(molecule)
    test_molecule("Ethanol", "CCO", labels, expected_charge=0)


if __name__ == "__main__":
    main()
