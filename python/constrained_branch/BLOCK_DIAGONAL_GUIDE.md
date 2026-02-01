# Block-Diagonal Constrained MPFIT: An Exhaustive Guide

This document explains every function in `_legacy_constrained_jax_pure.py` — what it does, why it exists, and how the pieces fit together. It is written so that an entry-level programmer can follow along, and so that they could then explain the ideas to a high-school student.

---

## Table of Contents

1. [The Problem in Plain English](#1-the-problem-in-plain-english)
2. [The Old Approach and Why It Broke](#2-the-old-approach-and-why-it-broke)
3. [The Block-Diagonal Fix](#3-the-block-diagonal-fix)
4. [Data Structures](#4-data-structures)
5. [Function-by-Function Walkthrough](#5-function-by-function-walkthrough)
6. [Solver Classes](#6-solver-classes)
7. [Memory Comparison](#7-memory-comparison)

---

## 1. The Problem in Plain English

Imagine you have a collection of molecules (like water, ethanol, acetone, etc.). Each molecule is made of atoms. Every atom carries some electric charge. We want to figure out what those charges are.

We don't measure charges directly. Instead, we have a "fingerprint" of each molecule called its **multipole moments** — numbers that describe how electric charge is distributed around each atom site. Our job is to find the set of charges that, when placed at atom positions, best reproduce those multipole fingerprints.

There's a constraint: **atoms of the same chemical type must get the same total charge.** For example, the three hydrogen atoms on a methyl group should all end up with the same charge, even if they're on different molecules.

This is an optimization problem: minimize the error between the multipole fingerprints we measured and the ones our charges would produce, subject to the equivalence constraint.

---

## 2. The Old Approach and Why It Broke

### The constraint matrix M

Instead of handling the constraint with complicated if/else logic during every optimization step, we encode it once as a matrix `M`.

Think of it this way:
- We have `n_params` **free parameters** (the independent numbers the optimizer can change).
- We have `n_atoms * n_sites` **charge slots** (every atom can contribute charge at every multipole site).
- `M` is a matrix that converts free parameters into charge slots: `charge_slots = M @ p0`.

The constraint (same-type atoms share charges) is baked into `M`. If atom 5 and atom 12 have the same type, their rows in `M` are linked so they always sum to the same total charge.

### The scaling problem

In the old code, `M` was a single dense matrix of shape `(N_total * N_total, n_params)` where `N_total` is the total number of atoms across ALL molecules.

| Molecules | Atoms (N) | M shape | M memory |
|-----------|-----------|---------|----------|
| 10 ethanol | 90 | (8100, ~2400) | 157 MB |
| 25 mixed | 225 | (50625, ~15000) | 6 GB |
| 50 mixed | 475 | (225625, ~60000) | 108 GB |

At 25+ molecules, the code ran out of memory and was killed by the OS.

### Why most of M was zeros

Here's the key insight: atom `i` on molecule A only contributes charge to sites on molecule A. It is physically too far away from molecule B's sites to matter. The `quse` matrix (which marks which atom-site pairs interact) is **block-diagonal** — it's only nonzero within each molecule's block.

```
quse for 3 molecules (schematic):

  [ X X X . . . . . . ]     X = nonzero (atom near site)
  [ X X X . . . . . . ]     . = zero (too far apart)
  [ X X X . . . . . . ]
  [ . . . X X X X . . ]
  [ . . . X X X X . . ]
  [ . . . X X X X . . ]
  [ . . . X X X X . . ]
  [ . . . . . . . X X ]
  [ . . . . . . . X X ]
```

The old code allocated the full dense matrix, including all the zeros. That's wasteful.

---

## 3. The Block-Diagonal Fix

Instead of one giant `M`, we build a **list of small M matrices**, one per molecule:

- `M_blocks[0]` has shape `(n_atoms_0 * n_atoms_0, n_params)` for molecule 0
- `M_blocks[1]` has shape `(n_atoms_1 * n_atoms_1, n_params)` for molecule 1
- etc.

Each small `M_i` maps the **same global parameter vector** `p0` to that molecule's charges. The atom-type equivalence constraint still works because atoms of the same type in different molecules reference the same columns of `p0`.

The objective function becomes a sum over molecules:

```
objective(p0) = sum_over_molecules( error_molecule_i(p0) ) + charge_penalty(p0)
```

Each molecule's error is computed independently using its own small arrays. No cross-molecule terms exist.

---

## 4. Data Structures

### `ConstrainedMPFITState`

A dataclass that holds everything about the system:

```python
@dataclass
class ConstrainedMPFITState:
    maxl: int = 4           # Maximum multipole rank (4 = hexadecapole)
    r1: float = 3.78        # Inner integration radius (bohr)
    r2: float = 9.45        # Outer integration radius (bohr)
    molecule_charge: float = 0.0  # Total system charge (e.g., 2.0 for N4444+P4444)
    conchg: float = 0.0     # Weight for charge constraint penalty

    atomtype: list[str]     # Atom type labels, e.g., ["C1", "H2", "H2", "O3", ...]
    quse: np.ndarray        # Binary mask: quse[site, atom] = 1 if atom is near site
    allcharge: np.ndarray   # Charge distribution: allcharge[site, atom]
    qstore: np.ndarray      # Total charge per atom: qstore[i] = sum(allcharge[:, i])
    multipoles: np.ndarray  # Reference multipole moments, shape (n_atoms, maxl+1, maxl+1, 2)
    xyzmult: np.ndarray     # Multipole site coordinates (bohr), shape (n_atoms, 3)
    xyzcharge: np.ndarray   # Charge site coordinates (bohr), shape (n_atoms, 3)
    lmax: np.ndarray        # Max multipole rank per site
    rvdw: np.ndarray        # Van der Waals radius per site (bohr)
    atom_counts: list[int]  # Number of atoms in each molecule, e.g., [9, 9, 5]
```

The `atom_counts` field is what enables the block-diagonal approach. It tells us where each molecule's atoms start and end in the global arrays.

---

## 5. Function-by-Function Walkthrough

### `setup_from_gdma_records(gdma_records, atom_type_labels) -> ConstrainedMPFITState`

**What it does:** Takes raw GDMA output (multipole moments from quantum chemistry) and atom type labels, and assembles them into a `ConstrainedMPFITState`.

**Line by line:**

```python
if not isinstance(gdma_records, list):
    gdma_records = [gdma_records]
```
Accept either a single record or a list. Wrap single records in a list for uniform processing.

```python
all_xyz = []          # Will hold coordinates for each molecule
all_multipoles = []   # Will hold multipole arrays for each molecule
all_rvdw = []         # Will hold VDW radii for each molecule
all_lmax = []         # Will hold max ranks for each molecule
atom_counts = []      # Will hold atom count for each molecule
total_atoms = 0       # Running total
```
Initialize empty lists. We'll append each molecule's data, then stack at the end.

```python
for gdma_record in gdma_records:
    molecule = Molecule.from_mapped_smiles(gdma_record.tagged_smiles, ...)
    n_atoms = molecule.n_atoms
    total_atoms += n_atoms
    atom_counts.append(n_atoms)
```
For each GDMA record: reconstruct the molecule object, count its atoms, record that count.

```python
    conformer_bohr = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
    all_xyz.append(conformer_bohr)
```
Convert atomic coordinates from angstroms to bohr (the atomic unit of length used in the math).

```python
    multipoles = _convert_flat_to_hierarchical(
        gdma_record.multipoles, n_atoms, gdma_settings.limit
    )
    all_multipoles.append(multipoles)
```
GDMA stores multipoles as flat arrays. Convert to hierarchical format: `[site, rank, component, real/imag]`.

```python
state.xyzcharge = np.vstack(all_xyz)   # Stack all molecules' coordinates
state.xyzmult = np.vstack(all_xyz)     # Same coordinates (sites = atoms)
state.multipoles = np.vstack(all_multipoles)
state.rvdw = np.concatenate(all_rvdw)
state.atom_counts = atom_counts        # <-- KEY for block-diagonal
state.quse = build_quse_matrix(...)    # Build the distance mask
```
Stack everything into global arrays. `atom_counts` records the boundaries.

---

### `build_quse_matrix(xyzmult, xyzcharge, rvdw) -> np.ndarray`

**What it does:** Creates a binary mask. `quse[s, i] = 1` means atom `i` is close enough to site `s` to contribute charge there.

```python
from scipy.spatial.distance import cdist
return (cdist(xyzmult, xyzcharge) < rvdw[:, None]).astype(int)
```

`cdist` computes all pairwise distances. We compare each distance to the VDW radius of that site. Because VDW radii are ~0.53 bohr but inter-molecular distances are ~10+ bohr, atoms from different molecules never pass this threshold. This is why `quse` is block-diagonal.

---

### `count_parameters(state) -> int`

**What it does:** Counts how many free parameters the optimizer will have, after accounting for atom-type equivalence constraints.

For each atom:
- If it's the **first atom of its type**, it gets one free parameter per active site.
- If it has a **twin** (an earlier atom of the same type), it gets one fewer free parameter (the last site's charge is determined by the constraint: "my total charge = twin's total charge").

```python
for i in range(n_atoms):
    n_sites_using = np.sum(quse[:, i])   # How many sites use this atom

    if i == 0:
        n_params += n_sites_using        # First atom: all sites are free
    else:
        twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
        if twin is not None:
            n_params += n_sites_using - 1   # One site determined by constraint
        else:
            n_params += n_sites_using       # New type: all sites are free
```

---

### `build_constraint_matrix(state) -> np.ndarray` (original, dense)

**What it does:** Builds the original dense `M` matrix of shape `(n_atoms * n_sites, n_params)`.

This is the OLD function, kept for reference and verification. It produces a single giant matrix.

The key logic for each atom `i`:

1. **First atom or new type:** Each active site gets its own column in `M`. Setting `M[i*n_sites + j, count] = 1.0` means "parameter `count` controls the charge of atom `i` at site `j`."

2. **Twin (same type as an earlier atom):** All active sites except the last get their own columns. The last active site's row is set to `twin_row_sum - earlier_row_sum`, which encodes:
   - `charge_at_last_site = total_charge_of_twin - charges_at_other_sites`
   - This ensures `qstore[i] == qstore[twin]` for any parameter vector `p0`.

---

### `build_constraint_matrices(state) -> list[np.ndarray]` (new, block-diagonal)

**What it does:** Same logic as `build_constraint_matrix`, but stores rows in per-molecule blocks instead of one global matrix.

**Key differences:**

```python
# Compute molecule offsets from atom_counts
offsets = []
s = 0
for c in atom_counts:
    offsets.append(s)
    s += c
```
If `atom_counts = [9, 9, 5]`, then `offsets = [0, 9, 18]`. Molecule 0's atoms are at global indices 0-8, molecule 1's at 9-17, molecule 2's at 18-22.

```python
# Map each global atom index to (molecule_index, local_atom_index)
atom_to_mol = {}
for mol_idx, (offset, n_atoms_mol) in enumerate(zip(offsets, atom_counts)):
    for local_i in range(n_atoms_mol):
        atom_to_mol[offset + local_i] = (mol_idx, local_i)
```
Example: global atom 11 → `atom_to_mol[11] = (1, 2)` means molecule 1, local atom 2.

```python
# Allocate per-molecule M blocks
M_blocks = [np.zeros((nc * nc, n_params)) for nc in atom_counts]
```
Each block is `(n_atoms_i^2, n_params)` — much smaller than `(N_total^2, n_params)`.

The iteration logic is the same as `build_constraint_matrix`, but:
- We only iterate over the molecule's own sites (`site_start:site_end`).
- We store in `M_blocks[mol_idx]` using local indices.
- For twin constraints across molecules, we look up the twin's rows from the twin's molecule block:

```python
twin_mol_idx, twin_local_i = atom_to_mol[twin]
twin_row_sum += M_blocks[twin_mol_idx][twin_local_i * twin_n_mol + twin_local_k, :]
```

This is the most subtle part: the twin might be in a different molecule, so we read from a different block.

---

### `kaisq_block(p0, M_blocks, mol_data, maxl, r1, r2, molecule_charge, conchg, xp=np)`

**What it does:** Computes the objective function (the number we're trying to minimize) as a sum of per-molecule contributions.

**Analogy:** Imagine grading a stack of homework papers. Each paper (molecule) gets its own score. The total class score is the sum of all paper scores, plus a penalty if the total "charge budget" doesn't add up.

**Line by line:**

```python
components = []
for l in range(maxl + 1):
    for m in range(l + 1):
        cs_range = [0] if m == 0 else [0, 1]
        for cs in cs_range:
            components.append((l, m, cs))
```
Build the list of all `(l, m, cs)` multipole components. For `maxl=4`, there are 25 components: `(0,0,0), (1,0,0), (1,1,0), (1,1,1), (2,0,0), ...`

These correspond to the "fingerprint channels" — monopole, dipole (x,y,z), quadrupole, etc.

```python
total_kai = 0.0      # Accumulated error
total_charge = 0.0   # Accumulated total charge

for M_i, (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) in zip(M_blocks, mol_data):
```
Loop over molecules. Each iteration handles one molecule independently.

```python
    n_atoms_i = xyzcharge_i.shape[0]
    n_sites_i = xyzmult_i.shape[0]  # == n_atoms_i (sites are at atom positions)

    allcharge_i = (xp.dot(M_i, p0)).reshape(n_atoms_i, n_sites_i).T
```
**This is the key line.** Multiply the molecule's constraint matrix by the global parameter vector. The result is a flat vector of length `n_atoms_i * n_sites_i`. Reshape it into a 2D charge matrix `(n_sites_i, n_atoms_i)` where `allcharge_i[site, atom]` is how much charge atom `atom` contributes at site `site`.

```python
    qstore_i = xp.sum(allcharge_i, axis=0)   # Total charge per atom
    total_charge = total_charge + xp.sum(qstore_i)
```
Sum over sites to get total charge per atom. Add to running total.

```python
    dx = xyzcharge_i[None, :, 0] - xyzmult_i[:, 0, None]  # shape: (n_sites_i, n_atoms_i)
    dy = xyzcharge_i[None, :, 1] - xyzmult_i[:, 1, None]
    dz = xyzcharge_i[None, :, 2] - xyzmult_i[:, 2, None]
```
Displacement vectors from each site to each atom. Broadcasting creates a 2D grid.

```python
    rvdw_arr = xp.asarray(rvdw_i, dtype=xp.float64)
    rmax = rvdw_arr + r2       # Outer integration shell boundary per site
    rminn = rvdw_arr + r1      # Inner integration shell boundary per site

    W = np.zeros((n_sites_i, maxl + 1))
    for ii in range(maxl + 1):
        W[:, ii] = (1.0 / (1.0 - 2.0 * ii)) * (rmax ** (1 - 2 * ii) - rminn ** (1 - 2 * ii))
```
Integration weights `W` from the MPFIT paper (J. Comp. Chem. Vol. 12, 913-917, 1991). These weight each multipole rank differently based on the integration shell geometry.

```python
    for l, m, cs in components:
        angular = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
        weight = angular * W[:, l]    # shape: (n_sites_i,)
```
Angular normalization factor times radial weight, per site.

```python
        rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz, xp=xp)  # (n_sites_i, n_atoms_i)
```
Evaluate regular solid harmonics — mathematical functions that describe how charge at position `(dx, dy, dz)` contributes to the multipole moment of rank `(l, m, cs)`.

```python
        sum1 = xp.sum(allcharge_i * rsh_vals, axis=1)  # (n_sites_i,)
```
For each site, sum up `charge * harmonic` over all atoms. This is the multipole moment that our charges would produce.

```python
        residuals = multipoles_i[:, l, m, cs] - sum1    # (n_sites_i,)
        total_kai = total_kai + xp.sum(weight * residuals ** 2)
```
Residual = measured multipole - predicted multipole. Square it, weight it, sum it. This is the error for this component.

```python
total_kai = total_kai + conchg * (total_charge - molecule_charge) ** 2
return total_kai
```
Add a penalty if the total charge doesn't match the expected molecular charge.

---

### `dkaisq_block(p0, M_blocks, mol_data, maxl, r1, r2, molecule_charge, conchg)`

**What it does:** Computes the **gradient** (derivative) of the objective function with respect to each parameter in `p0`. The optimizer uses this to know which direction to move parameters to reduce the error.

**Analogy:** If the objective function is "how far you are from the bottom of a valley," the gradient is "which direction is downhill, and how steep."

The structure mirrors `kaisq_block` exactly, but instead of accumulating error, it accumulates the derivative of error with respect to each charge, then maps back to parameters via `M_i.T`.

```python
grad = np.zeros(n_params)
total_charge = 0.0

for M_i, (multipoles_i, xyzmult_i, xyzcharge_i, rvdw_i) in zip(M_blocks, mol_data):
```
Same per-molecule loop.

```python
    d_allcharge_i = np.zeros((n_sites_i, n_atoms_i))
    for l, m, cs in components:
        ...
        coeff = 2.0 * weight * (multipoles_i[:, l, m, cs] - sum1)
        d_allcharge_i -= coeff[:, None] * rsh_vals
```
`d_allcharge_i[s, a]` = derivative of `total_kai` with respect to `allcharge_i[s, a]`. The minus sign and factor of 2 come from the chain rule on `(target - predicted)^2`.

```python
    d_flat_i = d_allcharge_i.T.flatten()
    grad += M_i.T @ d_flat_i
```
**Chain rule through M:** Since `allcharge_flat = M_i @ p0`, the derivative with respect to `p0` is `M_i.T @ d_allcharge_flat`. We accumulate across molecules because `p0` is shared.

```python
# Charge constraint gradient
charge_grad = conchg * 2.0 * (total_charge - molecule_charge)
for M_i in M_blocks:
    grad += charge_grad * M_i.T @ np.ones(M_i.shape[0])
```
The charge penalty's gradient: each charge slot contributes equally to total charge, so the gradient flows through `M_i.T @ ones`.

---

### `_slice_mol_data(state) -> list[tuple]`

**What it does:** Slices the global arrays into per-molecule pieces using `atom_counts`.

```python
mol_data = []
offset = 0
for n_atoms_i in state.atom_counts:
    mol_data.append((
        state.multipoles[offset:offset + n_atoms_i],   # This molecule's multipoles
        state.xyzmult[offset:offset + n_atoms_i],       # This molecule's site coords
        state.xyzcharge[offset:offset + n_atoms_i],     # This molecule's atom coords
        state.rvdw[offset:offset + n_atoms_i],           # This molecule's VDW radii
    ))
    offset += n_atoms_i
return mol_data
```

If `atom_counts = [9, 4, 5]`, molecule 0 gets rows 0:9, molecule 1 gets rows 9:13, molecule 2 gets rows 13:18.

---

### `expandcharge(p0, state)` (original, kept for reference)

The original imperative function that maps parameters to charges using if/else logic. `build_constraint_matrix` replaces this with a matrix multiply, and `build_constraint_matrices` does the same thing block-diagonally.

---

### `kaisq_pure` / `dkaisq_pure` (original dense versions)

The original vectorized objective and gradient that use a single dense `M` matrix. These work correctly but require allocating the full `(N^2, n_params)` matrix. Kept for verification on small problems.

---

### `kaisq` / `dkaisq` / `optimize_constrained` (original loop-based versions)

The very first implementation — loops over sites one by one, mutates state. Slowest but simplest to understand. Kept for reference.

---

## 6. Solver Classes

### `SciPySolver`

Uses numpy arrays and `scipy.optimize.minimize` with L-BFGS-B.

```python
class SciPySolver:
    def __init__(self, state):
        self.M_blocks = build_constraint_matrices(state)  # Build per-molecule M
        self.mol_data = _slice_mol_data(state)             # Slice per-molecule data
        self.n_params = count_parameters(state)

    def objective(self, p0):
        return kaisq_block(p0, self.M_blocks, self.mol_data, ...)

    def gradient(self, p0):
        return dkaisq_block(p0, self.M_blocks, self.mol_data, ...)

    def optimize(self, p0_init=None):
        result = minimize(fun=self.objective, x0=p0_init, jac=self.gradient,
                          method="L-BFGS-B", options={...})
        # Reconstruct qstore from per-molecule M blocks
        qstore_parts = []
        for M_i, (...) in zip(self.M_blocks, self.mol_data):
            allcharge_i = (M_i @ result.x).reshape(n_atoms_i, n_atoms_i).T
            qstore_parts.append(allcharge_i.sum(axis=0))
        qstore = np.concatenate(qstore_parts)
        return {"qstore": qstore, "objective": result.fun, ...}
```

### `JAXSolver`

Uses JAX arrays, JIT compilation, automatic differentiation, and `jaxopt.LBFGS`.

```python
class JAXSolver:
    def __init__(self, state):
        import jax, jax.numpy as jnp

        # GPU detection
        try:
            self._device = jax.devices("gpu")[0]
        except RuntimeError:
            self._device = jax.devices("cpu")[0]

        # Build M blocks and per-molecule data
        M_blocks_np = build_constraint_matrices(state)
        mol_data_np = _slice_mol_data(state)

        # Transfer to device (GPU or CPU)
        self._M_blocks = [jax.device_put(jnp.array(M_i), self._device) for M_i in M_blocks_np]
        self._mol_data = [
            (jax.device_put(jnp.array(mp), ...), jax.device_put(jnp.array(xm), ...), ...)
            for mp, xm, xc, rv in mol_data_np
        ]

        # JIT-compile the objective (closure captures device arrays)
        M_blocks = self._M_blocks
        mol_data = self._mol_data
        @jax.jit
        def objective(p0):
            return kaisq_block(p0, M_blocks, mol_data, ..., xp=jnp)
        self._objective = objective

        # Warmup: trigger compilation before timing
        _ = self._objective(jnp.zeros(self.n_params))

    def optimize(self, p0_init=None):
        solver = LBFGS(fun=self._objective, maxiter=1000, tol=1e-12, ...)
        res = solver.run(p0_init)
        # Reconstruct qstore same as SciPySolver
        ...
```

**Why `device_put`?** When arrays are explicitly placed on a device, JAX knows they're already there and doesn't embed them as constants in the compiled program (which would copy them every call).

**Why closure instead of class method?** `@jax.jit` on a method captures `self`, which JAX can't trace. Using a closure captures only the arrays and scalars that JAX needs.

**Why warmup?** JIT compilation happens on the first call. By calling with zeros during `__init__`, we move the compilation cost out of the timed optimization phase.

---

## 7. Memory Comparison

The dense M matrix has shape `(N_total^2, n_params)`. The block-diagonal approach replaces this with per-molecule blocks of shape `(n_i^2, n_params)`.

| Molecules | Atoms (N) | Dense M | Block-diagonal sum(M_i) | Reduction |
|-----------|-----------|---------|-------------------------|-----------|
| 1 ethanol (9 atoms) | 9 | 81 rows | 81 rows | 1x |
| 10 ethanol | 90 | 8,100 rows | 810 rows | 10x |
| 25 mixed (~9 atoms avg) | 225 | 50,625 rows | ~2,025 rows | 25x |
| 50 mixed | 475 | 225,625 rows | ~4,050 rows | ~56x |
| 100 mixed | 950 | 902,500 rows | ~8,100 rows | ~111x |

The reduction factor is approximately equal to the number of molecules (since cross-molecule blocks are all zeros in the dense version).

---

## Glossary

- **Multipole moment:** A mathematical description of charge distribution. Monopole = total charge. Dipole = charge separation. Quadrupole, octupole, hexadecapole = higher-order terms.
- **Regular solid harmonic:** A mathematical function `R_l^m(x,y,z)` used to project charge distributions onto multipole components.
- **GDMA:** Gaussian Distributed Multipole Analysis — a quantum chemistry method that computes multipole moments from electronic wavefunctions.
- **Atom type:** A label grouping chemically equivalent atoms. All atoms with the same label are constrained to have the same total charge.
- **Twin:** The first atom encountered with the same atom type label. Used to enforce the equivalence constraint.
- **`quse`:** Binary mask indicating which atom-site pairs interact (based on distance threshold).
- **`allcharge[s, i]`:** How much of atom `i`'s charge is "assigned" to site `s`.
- **`qstore[i]`:** Total charge of atom `i` = sum of `allcharge[:, i]`.
- **`p0`:** The reduced parameter vector — the free variables the optimizer adjusts.
- **L-BFGS-B / LBFGS:** A quasi-Newton optimization algorithm well-suited for smooth, quadratic-like objectives with many parameters.
- **JIT (Just-In-Time) compilation:** JAX traces your Python code once, compiles it to optimized machine code (XLA), then runs the compiled version on subsequent calls.
- **`device_put`:** Explicitly places an array on a specific device (CPU or GPU).
