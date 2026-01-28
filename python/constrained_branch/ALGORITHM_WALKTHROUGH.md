# Constrained MPFIT Algorithm Walkthrough

## Side-by-Side Dependency Trees

```
FORTRAN (mpfitroutines.f90)                 PYTHON (_legacy_constrained.py)
============================                ================================

┌─────────────────────────────┐             ┌─────────────────────────────┐
│  [0] variables.f90 MODULE   │             │  [0] ConstrainedMPFITState  │
│  ─────────────────────────  │             │  ─────────────────────────  │
│  Global state:              │             │  Dataclass with fields:     │
│  - atomtype(:)              │     ═══►    │  - atomtype: list[str]      │
│  - quse(:,:)                │             │  - quse: np.ndarray         │
│  - allcharge(:,:)           │             │  - allcharge: np.ndarray    │
│  - qstore(:)                │             │  - qstore: np.ndarray       │
│  - multipoles(:,:,:,:)      │             │  - multipoles: np.ndarray   │
│  - xyzmult(:,:)             │             │  - xyzmult: np.ndarray      │
│  - xyzcharge(:,:)           │             │  - xyzcharge: np.ndarray    │
│  - lmax(:), rvdw(:)         │             │  - lmax, rvdw: np.ndarray   │
│  - r1, r2 (parameters)      │             │  - r1, r2: float            │
└─────────────────────────────┘             └─────────────────────────────┘
              │                                           │
              ▼                                           ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│  [1] RSH(l,m,cs,xyz)        │             │  [1] rsh(l,m,cs,xyz)        │
│  ─────────────────────────  │             │  ─────────────────────────  │
│  Regular solid harmonics    │     ═══►    │  Regular solid harmonics    │
│  Lines 536-576              │             │  (identical math)           │
│  NO DEPENDENCIES            │             │  NO DEPENDENCIES            │
└─────────────────────────────┘             └─────────────────────────────┘

┌─────────────────────────────┐             ┌─────────────────────────────┐
│  [2] sitespecificdata()     │             │  [2] build_quse_matrix()    │
│  ─────────────────────────  │             │  ─────────────────────────  │
│  Build quse for ONE site    │     ═══►    │  Build quse for ALL sites   │
│  Lines 111-132              │             │  (vectorized version)       │
│  Uses: xyzmult, xyzcharge,  │             │  Uses: xyzmult, xyzcharge,  │
│        rvdw                 │             │        rvdw                 │
└─────────────────────────────┘             └─────────────────────────────┘

┌─────────────────────────────┐             ┌─────────────────────────────┐
│  [3] (implicit in           │             │  [3] count_parameters()     │
│       expandcharge)         │             │  ─────────────────────────  │
│  ─────────────────────────  │     ═══►    │  Count free params after    │
│  Parameter counting done    │             │  applying constraints       │
│  inline during expansion    │             │  Uses: atomtype, quse       │
└─────────────────────────────┘             └─────────────────────────────┘

              ┌───────────────────────────────────────────┐
              │     [4] expandcharge(p0)  ◄── THE CORE    │
              │     ─────────────────────────────────────  │
              │     Map reduced params → full charges     │
              │     ENFORCES CONSTRAINTS                  │
              │     Fortran lines 585-671                 │
              │     Uses: atomtype, quse                  │
              │     Modifies: allcharge, qstore           │
              └───────────────────────────────────────────┘
                        │                       │
          ┌─────────────┘                       └─────────────┐
          ▼                                                   ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│  [5] kaisq(p0)              │             │  [6] dkaisq(p0)             │
│  ─────────────────────────  │             │  ─────────────────────────  │
│  Objective function         │             │  Gradient of objective      │
│  Lines 212-344              │             │  Lines 352-527              │
│  Uses: [1] RSH              │             │  Uses: [1] RSH              │
│        [4] expandcharge     │             │        [4] expandcharge     │
│  Reads: multipoles, xyzmult │             │        [7] createdkaisq     │
│         xyzcharge, rvdw,    │             │  Reads: same as kaisq       │
│         lmax, r1, r2        │             │                             │
└─────────────────────────────┘             └─────────────────────────────┘
                                                          │
                                                          ▼
                                            ┌─────────────────────────────┐
                                            │  [7] createdkaisq(dparam1)  │
                                            │  ─────────────────────────  │
                                            │  Chain rule for constraints │
                                            │  Lines 674-773              │
                                            │  Uses: atomtype, quse       │
                                            │  Propagates gradients thru  │
                                            │  constraint dependencies    │
                                            └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  [8] OPTIMIZATION DRIVER                                                │
│  ───────────────────────                                                │
│  Fortran: frprmn() in frprmn.f90 (Conjugate Gradient)                   │
│  Python:  scipy.optimize.minimize(method='CG')                          │
│                                                                         │
│  Loop until converged:                                                  │
│      objective = kaisq(p0)      ──► calls [4] expandcharge              │
│      gradient  = dkaisq(p0)     ──► calls [4] expandcharge              │
│                                     calls [7] createdkaisq              │
│      p0 = update(p0, gradient)  ──► CG update rule                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Function-by-Function Education

### [0] State Variables (variables.f90 → ConstrainedMPFITState)

**Purpose:** Hold all data needed for the optimization.

**Fortran approach:** Global module variables, allocated once at program start.
```fortran
module variables
  Integer,parameter::maxl=4
  REAL*8,PARAMETER::r1=3.78,r2=9.45
  real*8,dimension(:),allocatable::qstore
  REAL*8,DIMENSION(:,:),ALLOCATABLE::allcharge
  character(3),dimension(:),allocatable::atomtype
  integer,dimension(:,:),allocatable::quse
  ! ... etc
end module variables
```

**Python approach:** Dataclass that encapsulates all state.
```python
@dataclass
class ConstrainedMPFITState:
    maxl: int = 4
    r1: float = 3.78
    r2: float = 9.45
    qstore: np.ndarray | None = None
    allcharge: np.ndarray | None = None
    atomtype: list[str] = field(default_factory=list)
    quse: np.ndarray | None = None
    # ... etc
```

**Key insight:** The Fortran uses `use variables` to access globals. Python passes
the state object explicitly (cleaner, more testable).

---

### [1] RSH - Regular Solid Harmonics

**Purpose:** Evaluate the multipole basis functions at a point.

**Mathematical definition:**
```
R_l^m(x,y,z) = r^l × Y_l^m(θ,φ)
```
where Y_l^m are spherical harmonics. We use real combinations:
- cs=0: cosine part (proportional to cos(mφ))
- cs=1: sine part (proportional to sin(mφ))

**Why needed:** The multipole moment from point charges is:
```
Q_l^m = Σ_j q_j × R_l^m(r_j - r_site)
```

**Fortran (lines 536-576):**
```fortran
FUNCTION RSH(l,m,cs,xyz)
  ! Hardcoded formulas for l=0,1,2,3,4
  rsharray(0,0,0)=1.0                           ! monopole
  rsharray(1,0,0)=z                             ! dipole z
  rsharray(1,1,0)=x                             ! dipole x
  rsharray(1,1,1)=y                             ! dipole y
  rsharray(2,0,0)=0.5*(3.0*z**2-rsq)            ! quadrupole
  ! ... etc for l=3,4
  RSH=rsharray(l,m,cs)
END FUNCTION RSH
```

**Python:** Identical math, just different syntax.

---

### [2] quse Matrix Construction

**Purpose:** Determine which atoms contribute to which multipole sites.

**The rule:** Atom j contributes to site i if:
```
distance(xyzmult[i], xyzcharge[j]) < rvdw[i]
```

**Physical meaning:** Only nearby atoms affect a site's local multipoles.

**Fortran (sitespecificdata, lines 111-132):**
```fortran
subroutine sitespecificdata
  ! Called once per site (site is a global variable)
  DO j=1,multsites
     rqm = distance(xyzmult(site,:), xyzcharge(j,:))
     IF (rqm < rvdw(site)) THEN
        quse(site,j)=1
     ELSE
        quse(site,j)=0
     ENDIF
  ENDDO
end subroutine
```

**Python:** Vectorized version that builds entire matrix at once.

---

### [3] Parameter Counting

**Purpose:** Determine how many free parameters we have after constraints.

**Without constraints:**
```
n_params = Σ_{i,s} quse[s,i]  (count all 1s in quse matrix)
```

**With constraints:**
```
For each atom i with a "twin" k (earlier atom, same atomtype):
    Subtract 1 from count (the last site is computed, not free)
```

**Example (water, atomtype = ["O", "H1", "H1"]):**
```
Atom 0 (O):  First of type "O"  → all quse sites are free
Atom 1 (H1): First of type "H1" → all quse sites are free
Atom 2 (H1): Twin is atom 1     → (n_sites - 1) are free, last is computed

If each atom uses 3 sites:
  Unconstrained: 3 + 3 + 3 = 9 params
  Constrained:   3 + 3 + 2 = 8 params (saved 1 param)
```

---

### [4] expandcharge - THE CORE CONSTRAINT MECHANISM ⭐

**Purpose:** Map reduced parameter vector p0 → full charges, enforcing constraints.

**This is the heart of the algorithm.** Everything else is standard least squares.

**The algorithm:**
```
count = 0  # index into p0

for i in range(n_atoms):
    # Step 1: Find twin (first earlier atom with same type)
    twin = None
    for k in range(i):
        if atomtype[k] == atomtype[i]:
            twin = k
            break

    # Step 2: Get sites that use this atom
    sites_using_i = [s for s in range(n_sites) if quse[s,i] == 1]
    n_sites_using = len(sites_using_i)

    charge_sum = 0.0

    if twin is None:
        # First occurrence of this type: ALL sites are free parameters
        for s in sites_using_i:
            allcharge[s, i] = p0[count]
            charge_sum += p0[count]
            count += 1
        qstore[i] = charge_sum

    else:
        # Has twin: use (n_sites - 1) free params, COMPUTE the last
        for j, s in enumerate(sites_using_i):
            if j < n_sites_using - 1:
                # Free parameter
                allcharge[s, i] = p0[count]
                charge_sum += p0[count]
                count += 1
            else:
                # LAST SITE: compute to enforce constraint
                allcharge[s, i] = qstore[twin] - charge_sum
                # Now: qstore[i] = charge_sum + (qstore[twin] - charge_sum)
                #                = qstore[twin]  ✓ CONSTRAINT ENFORCED!
        qstore[i] = qstore[twin]
```

**Why it works (mathematical proof):**
```
qstore[i] = Σ_s allcharge[s, i]           (definition)
          = Σ_{s < last} allcharge[s,i] + allcharge[last, i]
          = charge_sum + (qstore[twin] - charge_sum)
          = qstore[twin]  ✓
```

**Fortran (lines 585-671):** Same logic with 1-based indexing.

---

### [5] kaisq - Objective Function

**Purpose:** Compute the sum of squared multipole errors.

**The objective:**
```
J = Σ_sites Σ_{l,m} W_l × (Q_lm^GDMA - Q_lm^charges)²

where:
  Q_lm^GDMA    = GDMA multipole moment (input data)
  Q_lm^charges = Σ_j allcharge[site,j] × RSH(l,m,cs, r_j - r_site)
  W_l          = radial integration weight
```

**The radial weight W_l:**
```
W_l = (1/(1-2l)) × (rmax^(1-2l) - rmin^(1-2l))

where rmax = rvdw[site] + r2, rmin = rvdw[site] + r1
```

**Flow:**
```
kaisq(p0):
    expandcharge(p0)              # Get allcharge, qstore from params

    sumkai = 0
    for site in range(n_sites):
        for l in range(lmax[site]+1):
            for m in range(l+1):
                # Compute multipole from charges
                Q_charges = Σ_j allcharge[site,j] × RSH(l,m,cs, xyz_j - xyz_site)

                # Add squared error
                sumkai += W[l] × (multipoles[site,l,m,cs] - Q_charges)²

    return sumkai
```

---

### [6] dkaisq - Gradient of Objective

**Purpose:** Compute ∂kaisq/∂p0 for efficient optimization.

**Two stages:**

**Stage 1:** Compute ∂kaisq/∂allcharge[s,i] for all entries.
```
∂kaisq/∂allcharge[s,i] = -2 × W_l × (Q_lm^GDMA - Q_lm^charges) × RSH(l,m,cs, xyz_i - xyz_site)
```

**Stage 2:** Apply chain rule via createdkaisq() to get ∂kaisq/∂p0.

---

### [7] createdkaisq - Chain Rule for Constraints

**Purpose:** Convert ∂kaisq/∂allcharge → ∂kaisq/∂p0, accounting for constraint dependencies.

**The problem:** For constrained atom i with twin k:
```
allcharge[last, i] = qstore[k] - charge_sum
                   = (Σ_s allcharge[s,k]) - (Σ_{s<last} allcharge[s,i])
```

So allcharge[last,i] depends on:
- All of twin k's allcharge values (via qstore[k])
- All of i's earlier allcharge values (via charge_sum)

**The chain rule:**
```
∂kaisq/∂allcharge[s,k] += ∂kaisq/∂allcharge[last,i] × ∂allcharge[last,i]/∂qstore[k] × ∂qstore[k]/∂allcharge[s,k]
                        = ∂kaisq/∂allcharge[last,i] × 1 × 1
                        = ∂kaisq/∂allcharge[last,i]

∂kaisq/∂allcharge[s,i] -= ∂kaisq/∂allcharge[last,i]  (for s < last)
```

**The algorithm:**
```
createdkaisq(dparam1):  # dparam1 = ∂kaisq/∂allcharge in [atom,site] order

    # Step 1: Propagate constraint dependencies
    for i in range(1, n_atoms):
        twin = find_twin(i)
        if twin is not None:
            last_site = find_last_site(i)
            grad_last = dparam1[i * n_sites + last_site]

            # Add to all twin's entries
            for s in range(n_sites):
                dparam1[twin * n_sites + s] += grad_last

            # Subtract from i's earlier entries
            for s in range(last_site):
                dparam1[i * n_sites + s] -= grad_last

    # Step 2: Extract only free parameters (skip last site for constrained atoms)
    dkaisq = []
    for i in range(n_atoms):
        twin = find_twin(i)
        sites = get_sites_for_atom(i)

        if twin is None:
            # All sites are free
            for s in sites:
                dkaisq.append(dparam1[i * n_sites + s])
        else:
            # Skip last site (it's computed, not free)
            for s in sites[:-1]:
                dkaisq.append(dparam1[i * n_sites + s])

    return dkaisq
```

---

### [8] Optimization Driver

**Purpose:** Minimize kaisq using conjugate gradient with analytical gradients.

**Fortran:** Custom CG implementation in frprmn.f90 (Numerical Recipes style).

**Python:** scipy.optimize.minimize(method='CG').

```python
result = minimize(
    fun=lambda p: kaisq(p, state),
    x0=initial_guess,
    jac=lambda p: dkaisq(p, state),
    method='CG',
    options={'gtol': 1e-15, 'maxiter': 10000}
)
```

---

## Summary: The Constraint Magic

The entire constraint system boils down to **one clever trick in expandcharge()**:

```
For atom i with twin k (same atomtype):
    last_site_contribution = qstore[k] - sum_of_earlier_contributions
```

This single line enforces `qstore[i] == qstore[k]` **exactly**, not approximately.

Everything else (kaisq, dkaisq, createdkaisq) is just:
1. Standard least squares objective
2. Standard gradient computation
3. Chain rule to handle the constraint dependencies

---

## Implementation Order

```
[1] rsh()              ← Pure math, no dependencies
[2] build_quse_matrix  ← Uses coordinates only
[3] count_parameters   ← Uses atomtype, quse
[4] expandcharge       ← THE CORE - uses atomtype, quse; modifies allcharge, qstore
[5] kaisq              ← Uses [1], [4]; reads multipoles, coords
[7] createdkaisq       ← Uses atomtype, quse (chain rule logic)
[6] dkaisq             ← Uses [1], [4], [7]
[8] optimize + setup   ← Glue everything together
```
