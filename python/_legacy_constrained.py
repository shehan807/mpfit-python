"""
Legacy Constrained MPFIT Implementation
=======================================

This module is an educational, direct translation of the Fortran constrained MPFIT
code from `mpfit-python/f90/mpfit_source/source_constrain_fit/`.

The code structure mirrors the Fortran organization:

    Fortran File                    Python Equivalent
    ============                    =================
    variables.f90                   ConstrainedMPFITState (dataclass)
    mpfitroutines.f90 module        Setup helper functions
    expandcharge()                  expandcharge()
    kaisq()                         kaisq()
    dkaisq()                        dkaisq()
    createdkaisq()                  createdkaisq()
    RSH()                           rsh()

Call Flow Diagram
-----------------
The optimization follows this flow:

    1. Setup Phase:
       setup_from_gdma_record()
           ├── Initialize state (like Fortran ALLOCATE statements)
           ├── Build quse matrix (sitespecificdata logic)
           └── Set atomtype labels

    2. Optimization Phase (scipy.optimize.minimize):
       For each iteration:
           kaisq(p0)                    # Objective function
               └── expandcharge(p0)     # Map reduced params → full charges
                   └── Enforces: qstore[i] == qstore[twin] for same atomtype

           dkaisq(p0)                   # Gradient
               ├── expandcharge(p0)     # Recompute charges
               ├── Compute dparam       # ∂kaisq/∂allcharge[s,i]
               └── createdkaisq()       # Apply chain rule for constraints

    3. Result Extraction:
       expandcharge(optimal_p0)
       return qstore                    # Final per-atom charges

Constraint Mechanism (The Core Innovation)
------------------------------------------
For atoms with the same `atomtype` label:

    Standard (unconstrained):
        n_params = sum of quse[s,i] for all atoms i and sites s

    Constrained:
        For atom i with twin k (same atomtype, k < i):
            - Use (n_sites_for_i - 1) free parameters
            - Compute LAST site as: allcharge[last, i] = qstore[k] - sum(earlier)
            - This enforces: qstore[i] = qstore[k]

    n_params_constrained = n_params_unconstrained - (n_constrained_atoms)

References
----------
- Fortran source: mpfit-python/f90/mpfit_source/source_constrain_fit/
- Working prototype: library-charges/examples/constrained_mpfit_minimal.py
- MPFIT paper: JPC 1993, 97, 6628 (fitting radii r1=3.78, r2=9.45 Bohr)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from openff_pympfit.gdma.storage import MoleculeGDMARecord


# =============================================================================
# SECTION 1: GLOBAL STATE (mirrors variables.f90)
# =============================================================================
#
# Fortran uses a module with global variables. We encapsulate this in a dataclass
# that gets passed around (or used as module-level state for direct translation).
#
# Fortran variables.f90 declares:
#   - maxl: Maximum multipole rank (parameter, =4)
#   - r1, r2: Fitting shell radii (parameters)
#   - molecule_charge, conchg: Charge constraint parameters
#   - lmax(:): Max rank per site (allocatable)
#   - qstore(:): Total charge per atom (allocatable)
#   - xyzcharge(:,:): Charge coordinates (allocatable)
#   - xyzmult(:,:): Multipole coordinates (allocatable)
#   - allcharge(:,:): Per-site charge contributions (allocatable)
#   - multipoles(:,:,:,:): Multipole moments (allocatable)
#   - atomtype(:): Atom type labels (allocatable)
#   - quse(:,:): Binary mask for which atoms affect which sites (allocatable)
#   - rvdw(:): Van der Waals radii (allocatable)


@dataclass
class ConstrainedMPFITState:
    """
    Container for all state variables used in constrained MPFIT.

    This mirrors the Fortran `variables` module. All arrays are allocated
    during setup and modified during optimization.

    Fortran Reference: variables.f90, lines 1-44
    """

    # === Constants (Fortran PARAMETER statements) ===
    maxl: int = 4
    """Maximum multipole rank. Fortran: Integer,parameter::maxl=4 (line 11)"""

    r1: float = 3.78
    """Inner fitting radius in Bohr. Fortran: REAL*8,PARAMETER::r1=3.78 (line 21)"""

    r2: float = 9.45
    """Outer fitting radius in Bohr. Fortran: REAL*8,PARAMETER::r2=9.45 (line 21)"""

    molecule_charge: float = 0.0
    """Target total molecular charge. Fortran: molecule_charge=0.0 (line 28)"""

    conchg: float = 0.0
    """Weight for charge constraint penalty. 0 = no constraint. Fortran: conchg=0d0 (line 28)"""

    # === Allocatable arrays (set during setup) ===
    atomtype: list[str] = field(default_factory=list)
    """Atom type labels. Matching labels = constrained equal charges.
    Fortran: character(3),dimension(:),allocatable::atomtype (line 37)"""

    quse: np.ndarray | None = None
    """Binary mask [n_sites, n_atoms]. quse[s,i]=1 if atom i affects site s.
    Fortran: integer,dimension(:,:),allocatable::quse (line 38)"""

    allcharge: np.ndarray | None = None
    """Per-site charge contributions [n_sites, n_atoms].
    Fortran: REAL*8,DIMENSION(:,:),ALLOCATABLE::allcharge (line 35)"""

    qstore: np.ndarray | None = None
    """Total charge on each atom [n_atoms]. qstore[i] = sum(allcharge[:,i]).
    Fortran: real*8,dimension(:),allocatable::qstore (line 33)"""

    multipoles: np.ndarray | None = None
    """Multipole moments [n_sites, maxl+1, maxl+1, 2]. Last dim: 0=cos, 1=sin.
    Fortran: REAL*8,DIMENSION(:,:,:,:),ALLOCATABLE::multipoles (line 36)"""

    xyzmult: np.ndarray | None = None
    """Multipole site coordinates [n_sites, 3] in Bohr.
    Fortran: REAL*8,DIMENSION(:,:),ALLOCATABLE::xyzmult (line 35)"""

    xyzcharge: np.ndarray | None = None
    """Charge site coordinates [n_atoms, 3] in Bohr.
    Fortran: REAL*8,DIMENSION(:,:),ALLOCATABLE::xyzcharge (line 34)"""

    lmax: np.ndarray | None = None
    """Maximum multipole rank for each site [n_sites].
    Fortran: INTEGER,DIMENSION(:),ALLOCATABLE::lmax (line 32)"""

    rvdw: np.ndarray | None = None
    """Van der Waals radius for each site [n_sites] in Bohr.
    Fortran: real*8,dimension(:),allocatable::rvdw (line 39)"""


# Module-level state instance (for direct Fortran-style translation)
# In production code, this would be passed explicitly.
_state: ConstrainedMPFITState | None = None


# =============================================================================
# SECTION 2: SETUP FUNCTIONS (mirrors mpfitroutines.f90 module subroutines)
# =============================================================================
#
# These functions initialize the state from input data.
#
# Fortran mpfitroutines.f90 contains:
#   - numbersites(): Count sites from input file
#   - getmultmoments(): Read multipole moments
#   - gencharges(): Generate charge coordinates
#   - genvdw(): Generate van der Waals radii
#   - sitespecificdata(): Build quse matrix for each site


def setup_from_gdma_record(
    gdma_record: MoleculeGDMARecord,
    atom_type_labels: list[str],
) -> ConstrainedMPFITState:
    """
    Initialize state from a GDMA record.

    This combines the functionality of several Fortran subroutines:
    - numbersites() → inferred from gdma_record
    - getmultmoments() → from gdma_record.multipoles
    - gencharges() → from gdma_record.conformer
    - genvdw() → from gdma_settings.mpfit_atom_radius
    - sitespecificdata() → builds quse matrix

    Fortran Reference: mpfitroutines.f90, subroutines at lines 10-200

    Parameters
    ----------
    gdma_record : MoleculeGDMARecord
        GDMA data containing multipoles, conformer, and settings.
    atom_type_labels : list[str]
        Atom type labels specifying equivalence constraints.

    Returns
    -------
    ConstrainedMPFITState
        Initialized state ready for optimization.
    """
    from openff.toolkit import Molecule
    from openff.units import unit

    # Get molecule info
    molecule = Molecule.from_mapped_smiles(
        gdma_record.tagged_smiles, allow_undefined_stereo=True
    )
    n_atoms = molecule.n_atoms

    # Get GDMA settings
    gdma_settings = gdma_record.gdma_settings
    max_rank = gdma_settings.limit
    default_atom_radius = gdma_settings.mpfit_atom_radius

    # Create state object
    state = ConstrainedMPFITState()
    state.r1 = gdma_settings.mpfit_inner_radius
    state.r2 = gdma_settings.mpfit_outer_radius

    # Set atom type labels
    state.atomtype = atom_type_labels

    # Convert conformer coordinates to Bohr
    # Fortran: coordinates are in Bohr
    conformer_angstrom = gdma_record.conformer
    conformer_bohr = unit.convert(conformer_angstrom, unit.angstrom, unit.bohr)

    # Set coordinates (charge sites = multipole sites = atom positions)
    # Fortran: xyzcharge, xyzmult
    state.xyzcharge = conformer_bohr.copy()
    state.xyzmult = conformer_bohr.copy()

    # Set van der Waals radii
    # Fortran: genvdw() sets default, with special value for H
    state.rvdw = np.full(n_atoms, default_atom_radius)

    # Set lmax for each site
    # Fortran: lmax(:) from input file
    state.lmax = np.full(n_atoms, max_rank, dtype=float)

    # Convert flat multipoles to hierarchical format [n_sites, maxl+1, maxl+1, 2]
    # Fortran: multipoles(site, l, m, cs)
    from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical
    flat_multipoles = gdma_record.multipoles
    state.multipoles = _convert_flat_to_hierarchical(flat_multipoles, n_atoms, max_rank)

    # Build quse matrix
    # Fortran: sitespecificdata() called for each site
    state.quse = build_quse_matrix(state.xyzmult, state.xyzcharge, state.rvdw)

    # Initialize allcharge and qstore (will be populated by expandcharge)
    state.allcharge = np.zeros((n_atoms, n_atoms))
    state.qstore = np.zeros(n_atoms)

    return state


def build_quse_matrix(
    xyzmult: np.ndarray,
    xyzcharge: np.ndarray,
    rvdw: np.ndarray,
) -> np.ndarray:
    """
    Build the quse binary mask matrix.

    quse[s, i] = 1 if atom i contributes to the fit at multipole site s.
    The criterion is: distance(xyzmult[s], xyzcharge[i]) < rvdw[s]

    Fortran Reference: mpfitroutines.f90, sitespecificdata, lines 111-132

    Parameters
    ----------
    xyzmult : np.ndarray
        Multipole site coordinates [n_sites, 3]
    xyzcharge : np.ndarray
        Charge site coordinates [n_atoms, 3]
    rvdw : np.ndarray
        Van der Waals radii [n_sites]

    Returns
    -------
    np.ndarray
        Binary mask [n_sites, n_atoms], dtype=int
    """
    # =========================================================================
    # FORTRAN REFERENCE (mpfitroutines.f90, lines 111-132):
    # =========================================================================
    #
    # subroutine sitespecificdata
    #   use variables
    #   integer::i,j,multsites,midsites,count
    #   real*8::rqm
    #
    #   multsites=size(xyzmult(:,1));midsites=size(xyzcharge(:,1))-multsites
    #   i=site                          ! site is a global variable (current site)
    #   countatom=0
    #   countmid=0
    #   DO j=1,multsites
    #      rqm=((xyzmult(i,1)-xyzcharge(j,1))**2+(xyzmult(i,2)-xyzcharge(j,2))**2+&
    #           & (xyzmult(i,3)-xyzcharge(j,3))**2)**.5
    #      IF (rqm < rvdw(i)) THEN
    #         quse(site,j)=1
    #         countatom=countatom+1
    #      ELSE
    #         quse(site,j)=0
    #      ENDIF
    #   ENDDO
    # end subroutine sitespecificdata
    #
    # NOTE: Fortran calls this once per site. We vectorize for all sites at once.
    # =========================================================================

    n_sites = xyzmult.shape[0]
    n_atoms = xyzcharge.shape[0]

    # Fortran: quse(site,j) - we build entire matrix
    quse = np.zeros((n_sites, n_atoms), dtype=int)

    # Fortran loops: DO i=1,multsites (outer, over sites)
    #                DO j=1,multsites (inner, over atoms)
    for i in range(n_sites):  # Fortran: i=site (each site)
        for j in range(n_atoms):  # Fortran: j=1,multsites
            # Fortran: rqm=((xyzmult(i,1)-xyzcharge(j,1))**2 + ...)**0.5
            rqm = np.sqrt(
                (xyzmult[i, 0] - xyzcharge[j, 0]) ** 2
                + (xyzmult[i, 1] - xyzcharge[j, 1]) ** 2
                + (xyzmult[i, 2] - xyzcharge[j, 2]) ** 2
            )
            # Fortran: IF (rqm < rvdw(i)) THEN quse(site,j)=1
            if rqm < rvdw[i]:
                quse[i, j] = 1

    return quse


def count_parameters(state: ConstrainedMPFITState) -> int:
    """
    Count the number of reduced (free) parameters after applying constraints.

    Fortran Reference: Implicit in expandcharge logic, lines 585-671

    Parameters
    ----------
    state : ConstrainedMPFITState
        Current state with atomtype and quse set.

    Returns
    -------
    int
        Number of free parameters.
    """
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = quse.shape[0]

    n_params = 0

    for i in range(n_atoms):
        # Fortran: count1=0; DO j=1,multsites; if(quse(j,i).eq.1) count1=count1+1
        n_sites_using = 0
        for j in range(n_sites):
            if quse[j, i] == 1:
                n_sites_using += 1

        if i == 0:
            # Fortran: if(i.eq.1) then ... all sites are free
            n_params += n_sites_using
        else:
            # Fortran: twin=0; do k=1,i-1; if(atomtype(i).eq.atomtype(k)) twin=k
            twin = None
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            if twin is not None:
                # Has twin: last site is computed, not free
                n_params += n_sites_using - 1
            else:
                # First of this type: all sites are free
                n_params += n_sites_using

    return n_params


# =============================================================================
# SECTION 3: REGULAR SOLID HARMONICS (RSH function)
# =============================================================================
#
# Fortran: RSH(l, m, cs, xyz) at lines 536-576
#
# Evaluates regular solid harmonics R_l^m(x,y,z) for multipole expansion.
# These are the real-valued combinations of spherical harmonics:
#   - cs=0: cosine (real) part ~ cos(m*phi)
#   - cs=1: sine (imaginary) part ~ sin(m*phi)


def rsh(l: int, m: int, cs: int, xyz: np.ndarray) -> float:
    """
    Evaluate regular solid harmonic at point (x, y, z).

    Fortran Reference: mpfitroutines.f90, lines 536-576

    Parameters
    ----------
    l : int
        Angular momentum quantum number (0 <= l <= 4)
    m : int
        Magnetic quantum number (0 <= m <= l)
    cs : int
        0 for cosine (real), 1 for sine (imaginary)
    xyz : np.ndarray
        Cartesian coordinates [x, y, z]

    Returns
    -------
    float
        Value of R_l^m at xyz.
    """
    # Fortran: x=xyz(1);y=xyz(2);z=xyz(3)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Fortran: rsq=x**2+y**2+z**2
    rsq = x**2 + y**2 + z**2

    # Fortran: ALLOCATE(rsharray(0:4,0:4,0:1))
    rsharray = np.zeros((5, 5, 2))

    # l=0 (monopole)
    # Fortran: rsharray(0,0,0)=1.0
    rsharray[0, 0, 0] = 1.0

    # l=1 (dipole)
    # Fortran: rsharray(1,0,0)=z; rsharray(1,1,0)=x; rsharray(1,1,1)=y
    rsharray[1, 0, 0] = z
    rsharray[1, 1, 0] = x
    rsharray[1, 1, 1] = y

    # l=2 (quadrupole)
    # Fortran lines 551-555
    rsharray[2, 0, 0] = 0.5 * (3.0 * z**2 - rsq)
    rsharray[2, 1, 0] = np.sqrt(3.0) * x * z
    rsharray[2, 1, 1] = np.sqrt(3.0) * y * z
    rsharray[2, 2, 0] = 0.5 * np.sqrt(3.0) * (x**2 - y**2)
    rsharray[2, 2, 1] = np.sqrt(3.0) * x * y

    # l=3 (octupole)
    # Fortran lines 556-562
    rsharray[3, 0, 0] = 0.5 * (5.0 * z**3 - 3.0 * z * rsq)
    rsharray[3, 1, 0] = 0.25 * np.sqrt(6.0) * (4.0 * x * z**2 - x**3 - x * y**2)
    rsharray[3, 1, 1] = 0.25 * np.sqrt(6.0) * (4.0 * y * z**2 - y * x**2 - y**3)
    rsharray[3, 2, 0] = 0.5 * np.sqrt(15.0) * z * (x**2 - y**2)
    rsharray[3, 2, 1] = np.sqrt(15.0) * x * y * z
    rsharray[3, 3, 0] = 0.25 * np.sqrt(10.0) * (x**3 - 3.0 * x * y**2)
    rsharray[3, 3, 1] = 0.25 * np.sqrt(10.0) * (3.0 * x**2 * y - y**3)

    # l=4 (hexadecapole)
    # Fortran lines 564-572
    rsharray[4, 0, 0] = 0.125 * (
        8.0 * z**4 - 24.0 * (x**2 + y**2) * z**2 + 3.0 * (x**4 + 2.0 * x**2 * y**2 + y**4)
    )
    rsharray[4, 1, 0] = 0.25 * np.sqrt(10.0) * (4.0 * x * z**3 - 3.0 * x * z * (x**2 + y**2))
    rsharray[4, 1, 1] = 0.25 * np.sqrt(10.0) * (4.0 * y * z**3 - 3.0 * y * z * (x**2 + y**2))
    rsharray[4, 2, 0] = 0.25 * np.sqrt(5.0) * (x**2 - y**2) * (6.0 * z**2 - x**2 - y**2)
    rsharray[4, 2, 1] = 0.5 * np.sqrt(5.0) * x * y * (6.0 * z**2 - x**2 - y**2)
    rsharray[4, 3, 0] = 0.25 * np.sqrt(70.0) * z * (x**3 - 3.0 * x * y**2)
    rsharray[4, 3, 1] = 0.25 * np.sqrt(70.0) * z * (3.0 * x**2 * y - y**3)
    rsharray[4, 4, 0] = 0.125 * np.sqrt(35.0) * (x**4 - 6.0 * x**2 * y**2 + y**4)
    rsharray[4, 4, 1] = 0.5 * np.sqrt(35.0) * x * y * (x**2 - y**2)

    # Fortran: RSH=rsharray(l,m,cs)
    return rsharray[l, m, cs]


# =============================================================================
# SECTION 4: EXPANDCHARGE - THE CORE CONSTRAINT MECHANISM
# =============================================================================
#
# Fortran: expandcharge(p0) at lines 585-671
#
# This is the heart of the constraint system. It maps the reduced parameter
# vector p0 to the full allcharge matrix and qstore vector, enforcing that
# atoms with the same atomtype have identical total charges.


def expandcharge(p0: np.ndarray, state: ConstrainedMPFITState) -> None:
    """
    Map reduced parameters to full charges with atom-type constraints.

    THIS IS THE CORE CONSTRAINT MECHANISM.

    Fortran Reference: mpfitroutines.f90, expandcharge, lines 585-671

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector of length count_parameters().
    state : ConstrainedMPFITState
        State object (modified in place: allcharge and qstore updated).
    """
    # =========================================================================
    # FORTRAN REFERENCE (mpfitroutines.f90, lines 585-671):
    # =========================================================================
    #
    # subroutine expandcharge(p0)
    #   use variables
    #   real*8,dimension(:),intent(in)::p0
    #   integer::i,j,k,atoms,nmid,count,multsites,count1,count2,twin
    #   real*8::sum
    #
    #   multsites=size(xyzmult(:,1))
    #   atoms=size(atomtype)
    #   nmid=0
    #   allcharge=0.0
    #
    #   count=1
    #   do i=1,atoms
    #      ... [see inline comments below]
    #   enddo
    # end subroutine expandcharge
    # =========================================================================

    atomtype = state.atomtype
    quse = state.quse
    n_sites = state.xyzmult.shape[0]  # Fortran: multsites=size(xyzmult(:,1))
    n_atoms = len(atomtype)            # Fortran: atoms=size(atomtype)

    # Fortran: allcharge=0.0
    state.allcharge = np.zeros((n_sites, n_atoms))
    state.qstore = np.zeros(n_atoms)

    # Fortran: count=1 (1-based index into p0)
    count = 0  # Python: 0-based index into p0

    # Fortran: do i=1,atoms
    for i in range(n_atoms):
        # Fortran: count1=0; sum=0.0
        count1 = 0
        charge_sum = 0.0

        # =====================================================================
        # CASE 1: First atom (i == 0)
        # Fortran: if(i.eq.1) then
        # =====================================================================
        if i == 0:
            # All sites are free parameters
            # Fortran: do j=1,multsites; if(quse(j,i).eq.1) then
            for j in range(n_sites):
                if quse[j, i] == 1:
                    # Fortran: allcharge(j,i)=p0(count); sum=sum+p0(count); count=count+1
                    state.allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
            # Fortran: qstore(i)=sum
            state.qstore[i] = charge_sum

        # =====================================================================
        # CASE 2: Later atoms (i > 0)
        # Fortran: else
        # =====================================================================
        else:
            # --- Find twin (first earlier atom with same type) ---
            # Fortran: twin=0; do k=1,i-1; if(atomtype(i).eq.atomtype(k)) twin=k; enddo
            twin = None
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break  # Use first match (Fortran overwrites but first is what matters)

            # =================================================================
            # CASE 2a: Has twin - CONSTRAINED ATOM
            # Fortran: if(twin.ne.0) then
            # =================================================================
            if twin is not None:
                # Count how many sites use this atom
                # Fortran: do j=1,multsites; if(quse(j,i).eq.1) count1=count1+1; enddo
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        count1 += 1

                # Fill first (count1-1) sites from p0, compute LAST site
                # Fortran: count2=1
                count2 = 1
                # Fortran: do j=1,multsites
                for j in range(n_sites):
                    # Fortran: if((quse(j,i).eq.1).and.(count2<count1)) then
                    if quse[j, i] == 1 and count2 < count1:
                        # Free parameter
                        # Fortran: allcharge(j,i)=p0(count); sum=sum+p0(count); count=count+1; count2=count2+1
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                        count2 += 1
                    # Fortran: elseif((quse(j,i).eq.1).and.(count2.eq.count1)) then
                    elif quse[j, i] == 1 and count2 == count1:
                        # LAST SITE: compute to enforce constraint
                        # Fortran: allcharge(j,i)=qstore(twin)-sum
                        state.allcharge[j, i] = state.qstore[twin] - charge_sum
                        # Fortran: qstore(i)=qstore(twin)
                        state.qstore[i] = state.qstore[twin]
                        # ^^^ THE CONSTRAINT IS ENFORCED HERE ^^^

            # =================================================================
            # CASE 2b: No twin - first occurrence of this type
            # Fortran: else
            # =================================================================
            else:
                # All sites are free parameters (same as i==0 case)
                # Fortran: do j=1,multsites; if(quse(j,i).eq.1) then
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        # Fortran: allcharge(j,i)=p0(count); sum=sum+p0(count); count=count+1
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                # Fortran: qstore(i)=sum
                state.qstore[i] = charge_sum


# =============================================================================
# SECTION 5: OBJECTIVE FUNCTION (kaisq)
# =============================================================================
#
# Fortran: kaisq(p0) at lines 212-344
#
# The objective function computes the sum of squared errors between
# the GDMA multipoles and the multipoles generated by the fitted charges.


def kaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> float:
    """
    Objective function: sum of squared multipole errors.

    Fortran Reference: mpfitroutines.f90, kaisq function, lines 212-344

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector.
    state : ConstrainedMPFITState
        Current state with multipoles, coordinates, etc.

    Returns
    -------
    float
        Objective value (to be minimized).
    """
    # =========================================================================
    # FORTRAN REFERENCE (mpfitroutines.f90, lines 212-344):
    # =========================================================================
    #
    # function kaisq(p0)
    #   call expandcharge(p0)
    #   ...
    #   sumkai=0.0
    #   do s=1,multsites
    #      q0(:)=allcharge(s,:)
    #      ... compute W(i) for radial integration ...
    #      do l=0,lmax(s)
    #         do m=0,l
    #            sum1 = sum_j q0(j) * RSH(l,m,cs,xyz)
    #            sum = sum + W(l) * (multipoles(s,l,m,cs) - sum1)**2
    #         enddo
    #      enddo
    #      sumkai = sumkai + sum
    #   enddo
    #   ... add charge constraint ...
    #   kaisq = sumkai + sumcon
    # end function
    # =========================================================================

    # Fortran: call expandcharge(p0)
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]  # Fortran: multsites
    n_atoms = n_sites                  # Fortran: natom=multsites
    maxl = state.maxl

    # Fortran: sumkai=0.0
    sumkai = 0.0

    # Fortran: do s=1,multsites
    for s in range(n_sites):
        # Fortran: q0(:)=allcharge(s,:)
        q0 = state.allcharge[s, :]

        # Fortran: rmax=rvdw(site)+r2; rminn=rvdw(site)+r1
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        # Compute radial integration weights W[l]
        # Fortran: do i=0,lmax(s); W(i)=(1.0/(1.0-2.0*i))*(rmax**(1-2*i)-rminn**(1-2*i))
        W = np.zeros(maxl + 1)
        for i in range(int(state.lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        # Fortran: sum=0.0
        site_sum = 0.0

        # Fortran: do l=0,lmax(s)
        for l in range(int(state.lmax[s]) + 1):

            # Fortran: IF(l .EQ. 0) THEN
            if l == 0:
                # Monopole: only m=0, cs=0
                # Fortran: sum1=0.0; do j=1,natom; xyz=xyzqatom(j,:)-xyzmult(site,:); sum1=sum1+q0(j)*RSH(0,0,0,xyz)
                sum1 = 0.0
                for j in range(n_atoms):
                    xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                    sum1 += q0[j] * rsh(0, 0, 0, xyz)
                # Fortran: sum=(4.0*PI/(2.0*l+1.0))*W(0)*(multipoles(site,l,0,0)-sum1)**2
                site_sum = (4.0 * np.pi / (2.0 * l + 1.0)) * W[0] * (state.multipoles[s, l, 0, 0] - sum1) ** 2

            # Fortran: ELSE
            else:
                # Higher multipoles: m=0..l, cs=0,1
                # Fortran: DO m=0,l
                for m in range(l + 1):

                    # Fortran: IF(m .EQ. 0) THEN
                    if m == 0:
                        # m=0: only cosine part (cs=0)
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, 0, 0, xyz)
                        site_sum += (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, 0, 0] - sum1) ** 2

                    # Fortran: ELSE (m > 0)
                    else:
                        # m>0: both cosine (cs=0) and sine (cs=1) parts

                        # Cosine part (cs=0)
                        # Fortran: sum1=0.0; do j=1,natom; sum1=sum1+q0(j)*RSH(l,m,0,xyz)
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, m, 0, xyz)
                        site_sum += (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, 0] - sum1) ** 2

                        # Sine part (cs=1)
                        # Fortran: sum1=0.0; do j=1,natom; sum1=sum1+q0(j)*RSH(l,m,1,xyz)
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, m, 1, xyz)
                        site_sum += (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, 1] - sum1) ** 2

        # Fortran: sumkai=sumkai+sum
        sumkai += site_sum

    # =========================================================================
    # Add charge constraint penalty (usually conchg=0, so this does nothing)
    # Fortran: sumchg=0.0; do i=1,size(qstore); sumchg=sumchg+qstore(i); enddo
    # Fortran: sumcon=conchg*(sumchg-molecule_charge)**2
    # =========================================================================
    sumchg = np.sum(state.qstore)
    sumcon = state.conchg * (sumchg - state.molecule_charge) ** 2

    # Fortran: kaisq=sumkai+sumcon
    return sumkai + sumcon


# =============================================================================
# SECTION 6: GRADIENT FUNCTION (dkaisq)
# =============================================================================
#
# Fortran: dkaisq(p0) at lines 352-527
#
# Computes the analytical gradient of kaisq with respect to the reduced
# parameter vector p0. This is used for efficient CG optimization.


def dkaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """
    Gradient of kaisq with respect to reduced parameters p0.

    This is passed to scipy.optimize.minimize as jac=dkaisq.

    Fortran Reference: mpfitroutines.f90, dkaisq function, lines 352-527

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector.
    state : ConstrainedMPFITState
        Current state.

    Returns
    -------
    np.ndarray
        Gradient vector, same length as p0.
    """
    # Fortran: call expandcharge(p0)
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]  # Fortran: multsites
    n_atoms = n_sites                  # Fortran: natom=multsites
    npts = n_atoms                     # Fortran: npts
    maxl = state.maxl

    # Fortran: ALLOCATE(dparam(multsites*(natom+nmid)))
    # dparam stores ∂kaisq/∂allcharge in [site, atom] order
    dparam = np.zeros(n_sites * npts)

    # =========================================================================
    # Compute dparam = ∂kaisq/∂allcharge[s, j]
    # Fortran lines 395-501
    #
    # For kaisq = Σ W[l] * (Q_gdma - Q_charges)²
    # The derivative w.r.t. allcharge[s,j] is:
    #   ∂kaisq/∂allcharge[s,j] = -2 * W[l] * (Q_gdma - Q_charges) * ∂Q_charges/∂allcharge[s,j]
    #                          = -2 * W[l] * (Q_gdma - Q_charges) * RSH(l,m,cs, xyz_j - xyz_s)
    # =========================================================================

    # Fortran: do s=1,multsites
    for s in range(n_sites):
        # Fortran: q0(:)=allcharge(s,:)
        q0 = state.allcharge[s, :]

        # Fortran: rmax=rvdw(site)+r2; rminn=rvdw(site)+r1
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        # Fortran: do i=0,lmax(s); W(i)=...
        W = np.zeros(maxl + 1)
        for i in range(int(state.lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        # Fortran: do l=0,lmax(s)
        for l in range(int(state.lmax[s]) + 1):

            if l == 0:
                # Fortran: sum1=0.0; do j=1,natom; sum1+=q0(j)*RSH(0,0,0,xyz)
                sum1 = 0.0
                for j in range(n_atoms):
                    xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                    sum1 += q0[j] * rsh(0, 0, 0, xyz)

                # Fortran: sum=2.*(4.0*PI/(2.0*l+1.0))*W(0)*(multipoles(...)-sum1)
                coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[0] * (state.multipoles[s, l, 0, 0] - sum1)

                # Fortran: do j=1,natom; dparam((s-1)*npts+j) -= sum*RSH(...)
                for j in range(n_atoms):
                    xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                    dparam[s * npts + j] -= coeff * rsh(0, 0, 0, xyz)

            else:
                # Fortran: DO m=0,l
                for m in range(l + 1):

                    if m == 0:
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, 0, 0, xyz)

                        coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, 0, 0] - sum1)

                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * rsh(l, 0, 0, xyz)

                    else:
                        # Cosine part (cs=0)
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, m, 0, xyz)

                        coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, 0] - sum1)

                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * rsh(l, m, 0, xyz)

                        # Sine part (cs=1)
                        sum1 = 0.0
                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            sum1 += q0[j] * rsh(l, m, 1, xyz)

                        coeff = 2.0 * (4.0 * np.pi / (2.0 * l + 1.0)) * W[l] * (state.multipoles[s, l, m, 1] - sum1)

                        for j in range(n_atoms):
                            xyz = state.xyzcharge[j, :] - state.xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * rsh(l, m, 1, xyz)

    # =========================================================================
    # Reorder dparam from [site, atom] to [atom, site] → dparam1
    # Fortran lines 504-508: dparam1((j-1)*multsites+i)=dparam((i-1)*npts+j)
    # =========================================================================
    dparam1 = np.zeros(n_sites * npts)
    for i in range(n_sites):
        for j in range(npts):
            # Fortran: dparam1((j-1)*multsites+i) = dparam((i-1)*npts+j)
            dparam1[j * n_sites + i] = dparam[i * npts + j]

    # =========================================================================
    # Add charge constraint gradient (Fortran lines 511-515)
    # Usually conchg=0, so this does nothing
    # =========================================================================
    sumchg = np.sum(state.qstore)
    dparam1 += state.conchg * 2.0 * (sumchg - state.molecule_charge)

    # =========================================================================
    # Apply chain rule: convert ∂kaisq/∂allcharge → ∂kaisq/∂p0
    # Fortran: call createdkaisq(dkaisq, dparam1, atomtype, quse)
    # =========================================================================
    return createdkaisq(dparam1, state)


# =============================================================================
# SECTION 7: CHAIN RULE FOR CONSTRAINTS (createdkaisq)
# =============================================================================
#
# Fortran: createdkaisq(dkaisq, dparam1, atomtype, quse) at lines 674-773
#
# This applies the chain rule to convert gradients w.r.t. allcharge
# to gradients w.r.t. the reduced parameter vector p0.


def createdkaisq(
    dparam1: np.ndarray,
    state: ConstrainedMPFITState,
) -> np.ndarray:
    """
    Apply chain rule to convert full gradient to reduced parameter gradient.

    Fortran Reference: mpfitroutines.f90, createdkaisq, lines 674-773

    Parameters
    ----------
    dparam1 : np.ndarray
        Gradient w.r.t. allcharge in [atom, site] flattened order.
        dparam1[i * n_sites + j] = ∂kaisq/∂allcharge[j, i]
    state : ConstrainedMPFITState
        Current state with atomtype and quse.

    Returns
    -------
    np.ndarray
        Gradient w.r.t. reduced parameters p0.
    """
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = n_atoms  # Fortran: multsites = atoms

    # Fortran modifies in place; we copy
    dparam1 = dparam1.copy()

    # =========================================================================
    # PASS 1: Propagate constraint dependencies (Fortran lines 693-725)
    # =========================================================================
    # Fortran: do i=1,atoms
    for i in range(n_atoms):
        # Fortran: if(i.ne.1) then
        if i != 0:
            # Fortran: twin=0; do k=1,i-1; if(atomtype(i).eq.atomtype(k)) twin=k; goto 100
            twin = None
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            # Fortran: if(twin.ne.0) then
            if twin is not None:
                # Fortran: count1=0; do j=1,multsites; if(quse(j,i).eq.1) count1=count1+1
                count1 = 0
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        count1 += 1

                # Fortran: count2=1; do j=1,multsites
                count2 = 1
                for j in range(n_sites):
                    # Fortran: if((quse(j,i).eq.1).and.(count2.lt.count1)) count2=count2+1
                    if quse[j, i] == 1 and count2 < count1:
                        count2 += 1
                    # Fortran: elseif((quse(j,i).eq.1).and.(count2.eq.count1)) then
                    elif quse[j, i] == 1 and count2 == count1:
                        # j is the last site for atom i
                        # Fortran: do k=1,multsites; dparam1((twin-1)*multsites+k)+=dparam1((i-1)*multsites+j)
                        for k in range(n_sites):
                            dparam1[twin * n_sites + k] += dparam1[i * n_sites + j]
                        # Fortran: do k=1,j-1; dparam1((i-1)*multsites+k)-=dparam1((i-1)*multsites+j)
                        for k in range(j):
                            dparam1[i * n_sites + k] -= dparam1[i * n_sites + j]

    # =========================================================================
    # PASS 2: Extract free parameters (Fortran lines 729-770)
    # =========================================================================
    n_params = count_parameters(state)
    dkaisq_out = np.zeros(n_params)

    # Fortran: count=1
    count = 0

    # Fortran: do i=1,atoms
    for i in range(n_atoms):
        # Fortran: if(i.eq.1) then
        if i == 0:
            # Fortran: do j=1,multsites; if(quse(j,i).eq.1) then
            for j in range(n_sites):
                if quse[j, i] == 1:
                    dkaisq_out[count] = dparam1[i * n_sites + j]
                    count += 1
        else:
            # Fortran: twin=0; do k=1,i-1; if(atomtype(i).eq.atomtype(k)) twin=k
            twin = None
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            # Fortran: if(twin.ne.0) then
            if twin is not None:
                count1 = 0
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        count1 += 1

                # Fortran: count2=1
                count2 = 1
                for j in range(n_sites):
                    # Fortran: if((quse(j,i).eq.1).and.(count2.lt.count1)) then
                    if quse[j, i] == 1 and count2 < count1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count2 += 1
                        count += 1
                    # Last site skipped (not a free parameter)
            else:
                # Fortran: do j=1,multsites; if(quse(j,i).eq.1) then
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count += 1

    return dkaisq_out


# =============================================================================
# SECTION 8: OPTIMIZATION DRIVER
# =============================================================================


def optimize_constrained(
    state: ConstrainedMPFITState,
    p0_init: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run constrained MPFIT optimization.

    Uses scipy's conjugate gradient (CG) method with analytical gradients.
    The Fortran code uses a custom CG implementation (frprmn.f90), but
    scipy's CG achieves the same result.

    Fortran Reference: mpfit.f90 main program (calls frprmn for CG)

    Parameters
    ----------
    state : ConstrainedMPFITState
        Initialized state from setup_from_gdma_record().
    p0_init : np.ndarray, optional
        Initial guess for reduced parameters. Defaults to zeros.
    verbose : bool
        Whether to print optimization progress.

    Returns
    -------
    dict
        Optimization result with keys:
        - 'qstore': Final per-atom charges
        - 'allcharge': Final per-site contributions
        - 'objective': Final objective value
        - 'success': Whether optimization converged
        - 'scipy_result': Full scipy OptimizeResult
    """
    # Create initial guess if not provided
    n_params = count_parameters(state)
    if p0_init is None:
        p0_init = np.zeros(n_params)

    if verbose:
        print(f"Starting optimization with {n_params} free parameters")
        print(f"Initial objective: {kaisq(p0_init, state):.6e}")

    # Run scipy L-BFGS-B optimization with relaxed tolerance
    # Note: Python loops are ~100x slower than Fortran, so we use relaxed tolerances
    # for practical runtime. L-BFGS-B is faster than CG for this problem.
    result = minimize(
        fun=lambda p: kaisq(p, state),
        x0=p0_init,
        jac=lambda p: dkaisq(p, state),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-4},
    )

    if verbose:
        print(f"\nOptimization finished: {result.message}")
        print(f"Final objective: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")

    # Extract final charges by calling expandcharge with optimal params
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
    """
    Verify analytical gradient against numerical finite differences.
    (Debugging function - can be removed later)
    """
    analytical = dkaisq(p0, state)

    numerical = np.zeros_like(p0)
    for i in range(len(p0)):
        p_plus = p0.copy()
        p_minus = p0.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        numerical[i] = (kaisq(p_plus, state) - kaisq(p_minus, state)) / (2 * eps)

    max_rel_error = 0.0
    for i in range(len(p0)):
        if abs(numerical[i]) > 1e-10:
            rel_error = abs(analytical[i] - numerical[i]) / abs(numerical[i])
            max_rel_error = max(max_rel_error, rel_error)

    return analytical, numerical, max_rel_error


# =============================================================================
# SECTION 9: HIGH-LEVEL API
# =============================================================================


def generate_atom_type_labels_from_symmetry(
    molecule,
    equivalize_hydrogens: bool = True,
    equivalize_other_atoms: bool = True,
) -> list[str]:
    """
    Generate atom type labels based on molecular symmetry.

    This bridges the user-friendly `equivalize_*` flags to the Fortran-style
    atomtype labels used by the constrained solver.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule to analyze.
    equivalize_hydrogens : bool
        If True, topologically equivalent hydrogens get the same label.
    equivalize_other_atoms : bool
        If True, topologically equivalent heavy atoms get the same label.

    Returns
    -------
    list[str]
        Atom type labels like ["C0", "H1", "H1", "H1", "C2", "H3", "H3"]
    """
    from openff.recharge.utilities.toolkits import get_atom_symmetries
    from openff.units.elements import SYMBOLS

    # Get symmetry groups from OpenFF toolkit
    # Returns list like [0, 1, 1, 1, 1] for methane (C is group 0, all H are group 1)
    symmetry_groups = get_atom_symmetries(molecule)

    labels = []
    for i, (atom, group) in enumerate(zip(molecule.atoms, symmetry_groups)):
        element = SYMBOLS[atom.atomic_number]
        is_hydrogen = atom.atomic_number == 1

        # Check if we should use symmetry for this atom type
        if (is_hydrogen and equivalize_hydrogens) or (
            not is_hydrogen and equivalize_other_atoms
        ):
            # Use symmetry group → equivalent atoms get same label
            labels.append(f"{element}{group}")
        else:
            # Unique label for each atom → no constraint
            labels.append(f"{element}_{i}")

    return labels


def fit_constrained_mpfit(
    gdma_record: MoleculeGDMARecord,
    atom_type_labels: list[str] | None = None,
    equivalize_hydrogens: bool = True,
    equivalize_other_atoms: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    High-level API for constrained MPFIT fitting.

    This is the main entry point. It handles:
    1. Generating atom type labels (if not provided)
    2. Setting up the optimization state
    3. Running the optimization
    4. Returning the fitted charges

    Parameters
    ----------
    gdma_record : MoleculeGDMARecord
        GDMA data from Psi4 or other source.
    atom_type_labels : list[str], optional
        Explicit atom type labels. If provided, overrides equivalize_* flags.
    equivalize_hydrogens : bool
        If True and atom_type_labels not provided, equivalent H get same charge.
    equivalize_other_atoms : bool
        If True and atom_type_labels not provided, equivalent heavy atoms get same charge.
    verbose : bool
        Whether to print optimization progress.

    Returns
    -------
    np.ndarray
        Fitted charges, shape [n_atoms].
    """
    from openff.toolkit import Molecule

    # Get molecule for symmetry detection
    molecule = Molecule.from_mapped_smiles(
        gdma_record.tagged_smiles, allow_undefined_stereo=True
    )

    # Generate atom type labels if not provided
    if atom_type_labels is None:
        atom_type_labels = generate_atom_type_labels_from_symmetry(
            molecule,
            equivalize_hydrogens=equivalize_hydrogens,
            equivalize_other_atoms=equivalize_other_atoms,
        )

    if verbose:
        print(f"Atom type labels: {atom_type_labels}")

    # Setup state from GDMA record
    state = setup_from_gdma_record(gdma_record, atom_type_labels)

    # Run optimization
    result = optimize_constrained(state, verbose=verbose)

    return result["qstore"]


# =============================================================================
# SECTION 10: MAIN (for testing)
# =============================================================================


def generate_imidazolium_atom_types(molecule) -> list[str]:
    """
    Generate manual atom type labels for imidazolium cations.

    This enforces transferable charges:
    - Ring carbons C4/C5 share the same charge
    - Ring nitrogens share the same charge
    - Alkyl CH3 (terminal) groups share the same charge
    - Alkyl CH2 groups at same distance from ring share the same charge

    Works for EMIM, BMIM, C6MIM, etc.

    Imidazolium ring numbering:
    ```
           R1                R2
            \\               /
             N1-----C2-----N3(+)
              \\          //
               C5  ===  C4
    ```
    """
    from openff.units.elements import SYMBOLS

    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    # Find ring atoms using RDKit
    rdmol = molecule.to_rdkit()
    ring_info = rdmol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        if len(ring) == 5:
            ring_atoms.update(ring)

    # Identify ring atom types
    ring_nitrogens = []
    ring_carbons = []
    for i in ring_atoms:
        if molecule.atoms[i].atomic_number == 7:  # N
            ring_nitrogens.append(i)
        elif molecule.atoms[i].atomic_number == 6:  # C
            ring_carbons.append(i)

    # C2 is bonded to both nitrogens; C4/C5 are bonded to one N each
    c2_atom = None
    c45_atoms = []
    for c in ring_carbons:
        n_neighbors = sum(1 for b in molecule.atoms[c].bonds
                         if (b.atom1_index if b.atom2_index == c else b.atom2_index) in ring_nitrogens)
        if n_neighbors == 2:
            c2_atom = c
        else:
            c45_atoms.append(c)

    # Label ring atoms
    for n in ring_nitrogens:
        labels[n] = "N_ring"
    if c2_atom is not None:
        labels[c2_atom] = "C_ring_2"
    for c in c45_atoms:
        labels[c] = "C_ring_45"

    # Find alkyl carbons (not in ring)
    alkyl_carbons = [i for i in range(n_atoms)
                     if molecule.atoms[i].atomic_number == 6 and i not in ring_atoms]

    # Classify alkyl carbons by distance from ring and number of H
    for c in alkyl_carbons:
        # Count H neighbors
        h_count = sum(1 for b in molecule.atoms[c].bonds
                     if molecule.atoms[b.atom1_index if b.atom2_index == c else b.atom2_index].atomic_number == 1)

        if h_count == 3:
            labels[c] = "C_CH3"  # Terminal methyl
        elif h_count == 2:
            # CH2 - find distance from ring
            dist = _distance_from_ring(molecule, c, ring_atoms)
            labels[c] = f"C_CH2_{dist}"

    # Label hydrogens based on what they're attached to
    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:  # H
            # Find parent carbon/nitrogen
            parent = None
            for b in molecule.atoms[i].bonds:
                parent = b.atom1_index if b.atom2_index == i else b.atom2_index
                break

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
                else:
                    labels[i] = f"H_{parent}"

    return labels


def generate_quaternary_ammonium_atom_types(molecule) -> list[str]:
    """
    Generate manual atom type labels for quaternary ammonium/phosphonium cations.

    This enforces transferable charges for N4444, P4444, etc.:
    - Central N+/P+ is unique
    - All alpha carbons (bonded to N/P) are equivalent
    - All beta carbons are equivalent
    - All gamma carbons are equivalent
    - All terminal CH3 carbons are equivalent
    - Hydrogens follow their parent carbon type
    """
    from openff.units.elements import SYMBOLS

    n_atoms = molecule.n_atoms
    labels = [""] * n_atoms

    # Find central N+ or P+
    central_atom = None
    for i in range(n_atoms):
        atom = molecule.atoms[i]
        if atom.atomic_number in [7, 15]:  # N or P
            if atom.formal_charge == 1:
                central_atom = i
                break

    if central_atom is None:
        raise ValueError("Could not find central N+ or P+")

    # Label central atom
    element = "N" if molecule.atoms[central_atom].atomic_number == 7 else "P"
    labels[central_atom] = f"{element}_center"

    # BFS to find carbon distances from central atom
    carbon_distances = {}
    visited = {central_atom}
    queue = [(central_atom, 0)]

    while queue:
        current, dist = queue.pop(0)
        for b in molecule.atoms[current].bonds:
            neighbor = b.atom1_index if b.atom2_index == current else b.atom2_index
            if neighbor not in visited:
                visited.add(neighbor)
                if molecule.atoms[neighbor].atomic_number == 6:  # Carbon
                    carbon_distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                elif molecule.atoms[neighbor].atomic_number != 1:  # Not H
                    queue.append((neighbor, dist + 1))

    # Label carbons by distance (alpha=1, beta=2, gamma=3, delta/terminal=4)
    distance_names = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
    for c, dist in carbon_distances.items():
        # Check if terminal (has 3 H)
        h_count = sum(1 for b in molecule.atoms[c].bonds
                     if molecule.atoms[b.atom1_index if b.atom2_index == c else b.atom2_index].atomic_number == 1)
        if h_count == 3:
            labels[c] = "C_CH3"
        else:
            name = distance_names.get(dist, f"d{dist}")
            labels[c] = f"C_{name}"

    # Label hydrogens
    for i in range(n_atoms):
        if molecule.atoms[i].atomic_number == 1:
            parent = None
            for b in molecule.atoms[i].bonds:
                parent = b.atom1_index if b.atom2_index == i else b.atom2_index
                break
            if parent is not None and labels[parent]:
                # Extract carbon type and make H version
                c_type = labels[parent].replace("C_", "")
                labels[i] = f"H_{c_type}"

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

    return 99  # Not connected to ring


def get_manual_atom_types(molecule, mol_key: str) -> list[str] | None:
    """
    Get manual atom type labels for specific molecule types.

    Returns None if automatic symmetry detection should be used.
    """
    if mol_key in ["emim", "bmim", "c6mim"]:
        return generate_imidazolium_atom_types(molecule)
    elif mol_key in ["n4444", "p4444"]:
        return generate_quaternary_ammonium_atom_types(molecule)
    else:
        return None  # Use automatic symmetry detection


# Test molecule definitions
# Each entry: (name, smiles, expected_formal_charge)
TEST_MOLECULES = {
    "ethanol": (
        "Ethanol",
        "CCO",  # Simple SMILES - will be converted to mapped
        0,
    ),
    "emim": (
        "1-ethyl-3-methylimidazolium (EMIM)",
        "CCn1cc[n+](C)c1",  # EMIM cation
        1,
    ),
    "bmim": (
        "1-butyl-3-methylimidazolium (BMIM)",
        "CCCCn1cc[n+](C)c1",  # BMIM cation
        1,
    ),
    "c6mim": (
        "1-hexyl-3-methylimidazolium (C6MIM)",
        "CCCCCCn1cc[n+](C)c1",  # C6MIM cation
        1,
    ),
    "n4444": (
        "Tetrabutylammonium (N4444)",
        "CCCC[N+](CCCC)(CCCC)CCCC",  # N4444 cation
        1,
    ),
    "p4444": (
        "Tetrabutylphosphonium (P4444)",
        "CCCC[P+](CCCC)(CCCC)CCCC",  # P4444 cation
        1,
    ),
}


def test_molecule(
    name: str,
    smiles: str,
    expected_charge: int = 0,
    verbose: bool = True,
    verify_grad: bool = True,
    mol_key: str | None = None,
) -> dict:
    """
    Test constrained MPFIT on a single molecule.

    Parameters
    ----------
    name : str
        Human-readable molecule name for output.
    smiles : str
        SMILES string (will be converted to mapped SMILES).
    expected_charge : int
        Expected formal charge of the molecule.
    verbose : bool
        Whether to print detailed output.
    verify_grad : bool
        Whether to run gradient verification.

    Returns
    -------
    dict
        Test results including charges, constraints status, etc.
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

    # Create molecule from SMILES
    t_start = time.time()
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    timings["molecule_setup"] = time.time() - t_start

    if verbose:
        print(f"\nSMILES: {smiles}")
        print(f"Number of atoms: {molecule.n_atoms}")
        print(f"Formal charge: {molecule.total_charge}")
        print(f"  [Molecule setup: {timings['molecule_setup']:.2f}s ({timings['molecule_setup']/60:.2f} min)]")

    # Generate GDMA data using Psi4
    if verbose:
        print("\n--- Generating GDMA Data with Psi4 ---")
    gdma_settings = GDMASettings()
    if verbose:
        print(f"Method: {gdma_settings.method}, Basis: {gdma_settings.basis}")
        print("  [Running Psi4 (geometry opt + SCF + GDMA)...]")

    t_start = time.time()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=True
    )
    gdma_record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings
    )
    timings["psi4_gdma"] = time.time() - t_start

    if verbose:
        print(f"  [Psi4 GDMA complete: {timings['psi4_gdma']:.2f}s ({timings['psi4_gdma']/60:.2f} min)]")

    # Generate atom type labels - use manual types for ionic liquids, auto for others
    t_start = time.time()
    if mol_key is not None:
        labels = get_manual_atom_types(molecule, mol_key)
    else:
        labels = None

    if labels is None:
        labels = generate_atom_type_labels_from_symmetry(
            molecule, equivalize_hydrogens=True, equivalize_other_atoms=True
        )
        label_source = "auto-detected (symmetry)"
    else:
        label_source = "manual (transferable)"
    timings["atom_typing"] = time.time() - t_start

    if verbose:
        print("\n--- Atom Type Labels ---")
        print(f"Source: {label_source}")
        print(f"Labels: {labels}")
        print(f"  [Atom typing: {timings['atom_typing']:.2f}s ({timings['atom_typing']/60:.2f} min)]")

    # Show which atoms are equivalent
    equiv_classes = {}
    for i, label in enumerate(labels):
        if label not in equiv_classes:
            equiv_classes[label] = []
        equiv_classes[label].append(i)

    if verbose:
        print("\nEquivalence classes:")
        for label, indices in equiv_classes.items():
            atoms_str = ", ".join(
                f"{i}({SYMBOLS[molecule.atoms[i].atomic_number]})" for i in indices
            )
            print(f"  {label}: [{atoms_str}]")

    # Setup and count parameters
    t_start = time.time()
    state = setup_from_gdma_record(gdma_record, labels)
    state.molecule_charge = float(expected_charge)

    n_full = int(np.sum(state.quse))
    n_reduced = count_parameters(state)
    timings["mpfit_setup"] = time.time() - t_start

    if verbose:
        print("\n--- Setup ---")
        print(f"Full parameters (unconstrained): {n_full}")
        print(f"Reduced parameters (constrained): {n_reduced}")
        print(f"Parameters saved: {n_full - n_reduced}")
        print(f"  [MPFIT setup: {timings['mpfit_setup']:.2f}s ({timings['mpfit_setup']/60:.2f} min)]")

    # Verify gradient before optimization
    if verify_grad:
        if verbose:
            print("\n--- Gradient Verification ---")
        t_start = time.time()
        p0_test = np.random.randn(n_reduced) * 0.01
        _, _, max_error = verify_gradient(p0_test, state)
        timings["gradient_verify"] = time.time() - t_start
        if verbose:
            print(f"Max relative error: {max_error:.2e}")
            print(f"Gradient check: {'PASS' if max_error < 1e-4 else 'FAIL'}")
            print(f"  [Gradient verification: {timings['gradient_verify']:.2f}s ({timings['gradient_verify']/60:.2f} min)]")

    # Run optimization
    if verbose:
        print("\n--- Optimization ---")
    t_start = time.time()
    result = optimize_constrained(state, verbose=verbose)
    timings["optimization"] = time.time() - t_start

    if verbose:
        print(f"  [Optimization: {timings['optimization']:.2f}s ({timings['optimization']/60:.2f} min)]")

    # Show results
    if verbose:
        print("\n--- Final Charges ---")
        for i, (label, q) in enumerate(zip(labels, result["qstore"])):
            element = SYMBOLS[molecule.atoms[i].atomic_number]
            print(f"  Atom {i:2d} ({element:2s}, type={label:4s}): {q:+.6f}")

    # Verify constraints
    all_satisfied = True
    constraint_results = {}
    for label, indices in equiv_classes.items():
        if len(indices) > 1:
            charges = [result["qstore"][i] for i in indices]
            max_diff = max(charges) - min(charges)
            satisfied = max_diff < 1e-10
            if not satisfied:
                all_satisfied = False
            constraint_results[label] = {
                "indices": indices,
                "charges": charges,
                "max_diff": max_diff,
                "satisfied": satisfied,
            }

    if verbose:
        print("\n--- Constraint Verification ---")
        for label, info in constraint_results.items():
            status = "PASS" if info["satisfied"] else "FAIL"
            print(f"  {label}: max difference = {info['max_diff']:.2e} [{status}]")

        total_charge = np.sum(result["qstore"])
        print(f"\nTotal molecular charge: {total_charge:.6f}")
        print(f"Expected charge: {expected_charge}")
        print(f"All constraints satisfied: {'YES' if all_satisfied else 'NO'}")

    # Total timing
    timings["total"] = time.time() - t_total_start

    if verbose:
        print("\n--- Timing Summary ---")
        print(f"  Molecule setup:       {timings['molecule_setup']:6.2f}s ({timings['molecule_setup']/60:5.2f} min)")
        print(f"  Psi4 GDMA:            {timings['psi4_gdma']:6.2f}s ({timings['psi4_gdma']/60:5.2f} min)")
        print(f"  Atom typing:          {timings['atom_typing']:6.2f}s ({timings['atom_typing']/60:5.2f} min)")
        print(f"  MPFIT setup:          {timings['mpfit_setup']:6.2f}s ({timings['mpfit_setup']/60:5.2f} min)")
        if verify_grad:
            print(f"  Gradient verification:{timings['gradient_verify']:6.2f}s ({timings['gradient_verify']/60:5.2f} min)")
        print(f"  Optimization:         {timings['optimization']:6.2f}s ({timings['optimization']/60:5.2f} min)")
        print(f"  ----------------------------------------")
        print(f"  TOTAL:                {timings['total']:6.2f}s ({timings['total']/60:5.2f} min)")

    return {
        "name": name,
        "smiles": smiles,
        "n_atoms": molecule.n_atoms,
        "n_params_full": n_full,
        "n_params_reduced": n_reduced,
        "qstore": result["qstore"],
        "labels": labels,
        "equiv_classes": equiv_classes,
        "constraint_results": constraint_results,
        "all_satisfied": all_satisfied,
        "total_charge": np.sum(result["qstore"]),
        "expected_charge": expected_charge,
        "objective": result["objective"],
        "timings": timings,
    }


def main():
    """
    Test the constrained MPFIT implementation with multiple molecules.

    Usage:
        python -m openff_pympfit.mpfit._legacy_constrained [molecule_key]

    If molecule_key is provided, only that molecule is tested.
    Otherwise, ethanol is tested by default.

    Available molecules:
        ethanol, emim, bmim, c6mim, n4444, p4444
    """
    import sys

    print("=" * 70)
    print("Legacy Constrained MPFIT - Educational Implementation")
    print("=" * 70)

    print("\nCall flow diagram:")
    print("""
    fit_constrained_mpfit(gdma_record, atom_type_labels)
        │
        ├─► generate_atom_type_labels_from_symmetry()  [if labels not provided]
        │
        ├─► setup_from_gdma_record()
        │       ├── Convert coordinates to Bohr
        │       ├── Build multipoles array
        │       └── build_quse_matrix()
        │
        └─► optimize_constrained()
                │
                └─► scipy.optimize.minimize(kaisq, jac=dkaisq, method='CG')
                        │
                        ├─► kaisq(p0)
                        │       └── expandcharge(p0)  ◄── THE CONSTRAINT MAGIC
                        │               └── Enforces qstore[i] == qstore[twin]
                        │
                        └─► dkaisq(p0)
                                ├── expandcharge(p0)
                                ├── Compute ∂kaisq/∂allcharge
                                └── createdkaisq()  ◄── CHAIN RULE FOR CONSTRAINTS
    """)

    # Determine which molecule(s) to test
    if len(sys.argv) > 1:
        molecule_key = sys.argv[1].lower()
        if molecule_key == "all":
            molecules_to_test = list(TEST_MOLECULES.keys())
        elif molecule_key in TEST_MOLECULES:
            molecules_to_test = [molecule_key]
        else:
            print(f"\nUnknown molecule: {molecule_key}")
            print(f"Available: {', '.join(TEST_MOLECULES.keys())}, all")
            sys.exit(1)
    else:
        molecules_to_test = ["ethanol"]

    print(f"\nMolecules to test: {molecules_to_test}")

    # Run tests
    results = {}
    for mol_key in molecules_to_test:
        name, smiles, expected_charge = TEST_MOLECULES[mol_key]
        try:
            results[mol_key] = test_molecule(
                name, smiles, expected_charge, verbose=True, verify_grad=True, mol_key=mol_key
            )
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()
            results[mol_key] = {"error": str(e)}

    # Summary
    if len(molecules_to_test) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for mol_key, result in results.items():
            if "error" in result:
                print(f"  {mol_key}: ERROR - {result['error']}")
            else:
                status = "PASS" if result["all_satisfied"] else "FAIL"
                print(
                    f"  {mol_key}: {result['n_atoms']} atoms, "
                    f"{result['n_params_reduced']}/{result['n_params_full']} params, "
                    f"q_total={result['total_charge']:.4f}, [{status}]"
                )

    print("\nImplementation complete! All core functions are now implemented:")
    print("  [1] rsh()                    - Regular solid harmonics")
    print("  [2] build_quse_matrix()      - Which atoms affect which sites")
    print("  [3] count_parameters()       - Count free parameters")
    print("  [4] expandcharge()           - THE CORE constraint mechanism")
    print("  [5] kaisq()                  - Objective function")
    print("  [6] dkaisq()                 - Gradient (for scipy jac=)")
    print("  [7] createdkaisq()           - Chain rule for constraints")
    print("  [8] setup/optimize functions - High-level API")


if __name__ == "__main__":
    main()
