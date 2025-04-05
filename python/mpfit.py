import sys
import numpy as np 

def _print_multipole_moments(i, mm, lmax):
    """
    Print multipole moments for site i in a format similar to the original file

    Parameters:
    ----------
    i : int
        Site index
    mm : ndarray
        4D array containing multipole moments
    lmax : int
        Maximum rank for this site
    """
    # Print monopole
    print(f"                   Q00  =  {mm[i, 0, 0, 0]:10.6f}")

    # Print higher order multipoles if present
    for l in range(1, lmax[i] + 1):
        # Calculate and print |Ql|
        q_norm_squared = mm[i, l, 0, 0]**2
        for j in range(1, l + 1):
            q_norm_squared += mm[i, l, j, 0]**2 + mm[i, l, j, 1]**2
        q_norm = np.sqrt(q_norm_squared)

        print(f"|Q{l}| = {q_norm:10.6f}  Q{l}0  = {mm[i, l, 0, 0]:10.6f}", end="")

        # Print components
        for j in range(1, l + 1):
            if j == 1:
                print(f"  Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}", end="")
            else:
                # For j > 1, print on new line with spacing
                if j == 2:
                    print()
                print(f"                   Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}", end="")
        print()

def numbersites(inpfile):
    count = 0
    with open(inpfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            if len(line_split) >= 5:
                _type = line_split[0]
                x, y, z = map(float, line_split[1:4])
                maxl = int(line_split[4])
                for i in range(maxl + 1):
                    skip_lines = f.readline()
                count += 1
    return count

def getmultmoments(
    inpfile,
    n, 
    lmax, 
    mm, # multipole moments
    ms, # multipole sites 
    atomtype,
    reprint_mm=False, 
    ):
    with open(inpfile, 'r') as f:
        for i in range(n):
            line = f.readline().split()
            atomtype[i] = line[0]
            x, y, z = float(line[1]), float(line[2]), float(line[3])
            lmax[i] = int(line[4])
            
            ms[i,0] = x; ms[i,1] = y; ms[i,2] = z;

            q0 = float(f.readline().strip()) # monopole
            mm[i, 0, 0, 0] = q0

            if lmax[i] > 0: 
                for l in range(1, lmax[i] + 1):
                    line = f.readline().split()
                    mm[i, l, 0, 0] = float(line[0]) # Q_l0
                    for m in range(1, l + 1): # Q_lm (m>0)
                        idx = 2*m - 1
                        mm[i, l, m, 0] = float(line[idx]) # real
                        mm[i, l, m, 1] = float(line[idx+1]) # imaginary
        
        if reprint_mm:
            # After the with open block:
            for i in range(n):
                print(f"Site {i+1}:")
                _print_multipole_moments(i, mm, lmax)
                print()
    return lmax, mm, ms, atomtype

def gencharges(ms, qs, midbond):
    """Generate charge positions from multipole sites and bond information"""
    nmult = ms.shape[0] # number of multipole sites
    nmid = qs.shape[0] - nmult # number of midpoints 

    # copy multipole site coordinates to charge sites
    for i in range(nmult):
        qs[i, 0] = ms[i, 0]
        qs[i, 1] = ms[i, 1]
        qs[i, 2] = ms[i, 2]
    
    if (nmid > 0):
        count = 0
        for i in range(nmult):
            for j in range(i+1, nmult):
                if midbond[i, j] == 1:
                    # add a midpoint charge 
                    qs[nmult + count, 0] = (ms[i, 0] + ms[j, 0]) / 2.0
                    qs[nmult + count, 1] = (ms[i, 1] + ms[j, 1]) / 2.0
                    qs[nmult + count, 2] = (ms[i, 2] + ms[j, 2]) / 2.0
                    count += 1

    return qs

def Amat(nsite, xyzmult, xyzcharge, r1, r2, maxl, A):
    """Construct A matrix as in J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)

    Returns 3D array A(i,j,k) where i stands for the specific multipole, 
    j,k for the charges
    """
    ncharge = xyzcharge.shape[0] # or len(xyzcharge)

    W = np.zeros(maxl + 1)
    # compute W integration factor
    for i in range(maxl):
        W[i] = (1.0 / (1.0 - 2.0*i)) * (r2**(1-2*i) - r1**(1-2*i))

    for j in range(ncharge):
        # Position relative to multipole site
        xj = xyzcharge[j, 0] - xyzmult[nsite, 0]
        yj = xyzcharge[j, 1] - xyzmult[nsite, 1]
        zj = xyzcharge[j, 2] - xyzmult[nsite, 2]
        for k in range(ncharge):
            # Position relative to multipole site
            xk = xyzcharge[k, 0] - xyzmult[nsite, 0]
            yk = xyzcharge[k, 1] - xyzmult[nsite, 1]
            zk = xyzcharge[k, 2] - xyzmult[nsite, 2]

            _sum = 0.0
            for l in range(0, maxl):
                if l == 0:
                    _sum = (1.0 / (2.0*l + 1.0)) * W[0] * RSH(0, 0, 0, xj, yj, zj) * RSH(0, 0, 0, xk, yk, zk)
                else:
                    for m in range(l):
                        if m == 0:
                            _sum += (1.0 / (2.0*l + 1.0)) * W[l] * (RSH(l, 0, 0, xj, yj, zj) * RSH(l, 0, 0, xk, yk, zk))
                        else:
                            # For m>0, include both real and imaginary parts
                            _sum += (1.0 / (2.0*l + 1.0)) * W[l] * (
                                RSH(l, m, 0, xj, yj, zj) * RSH(l, m, 0, xk, yk, zk) +
                                RSH(l, m, 1, xj, yj, zj) * RSH(l, m, 1, xk, yk, zk)
                            )
            A[j, k] = _sum
    return A


def bvec(nsite, xyzmult, xyzcharge, r1, r2, maxl, multipoles, b):
    """Construct b vector as in  J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)"""
    ncharge = xyzcharge.shape[0]

    W = np.zeros(maxl+1, dtype=np.float64)
    for i in range(maxl+1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2**(1 - 2 * i) - r1**(1 - 2 * i))
    for k in range(ncharge):
        # Compute relative coordinates
        xk = xyzcharge[k, 0] - xyzmult[nsite, 0]
        yk = xyzcharge[k, 1] - xyzmult[nsite, 1]
        zk = xyzcharge[k, 2] - xyzmult[nsite, 2]

        _sum = 0.0
        for l in range(maxl+1):
            if l == 0:
                # Special case for l = 0
                _sum = (1.0 / (2.0 * l + 1.0)) * W[0] * \
                        multipoles[nsite, 0, 0, 0] * RSH(0, 0, 0, xk, yk, zk)
            else:
                for m in range(l+1):
                    if m == 0: 
                        # m = 0 case
                        _sum += (1.0 / (2.0 * l + 1.0)) * W[l] * \
                                multipoles[nsite, l, 0, 0] * RSH(l, 0, 0, xk, yk, zk)
                    else:
                        # m > 0 case
                        _sum += (1.0 / (2.0 * l + 1.0)) * W[l] * \
                                   (multipoles[nsite, l, m, 0] * RSH(l, m, 0, xk, yk, zk) +
                                    multipoles[nsite, l, m, 1] * RSH(l, m, 1, xk, yk, zk))
        b[k] = _sum
    return b

def RSH(l,m,cs,x,y,z):
    """Evaluate regular spherical harmonics.
    
    Evaluate spherical harmonics @ x,y,z, where l,m,cs determine 
    the rank, cs=0 means cosine, cs=1 means sine.
    """
    # initialize array for RSH values
    rsharray = np.zeros((5, 5, 2)) # ALLOCATE(rsharray(0:4,0:4,0:1))
    rsq = x**2 + y**2 + z**2 

    # l=0 (monopole)
    rsharray[0, 0, 0] = 1.0
    
    # l=1 (dipole)
    rsharray[1, 0, 0] = z
    rsharray[1, 1, 0] = x
    rsharray[1, 1, 1] = y
    
    # l=2 (quadrupole)
    rsharray[2, 0, 0] = 0.5 * (3.0 * z**2 - rsq)
    rsharray[2, 1, 0] = np.sqrt(3.0) * x * z
    rsharray[2, 1, 1] = np.sqrt(3.0) * y * z
    rsharray[2, 2, 0] = 0.5 * np.sqrt(3.0) * (x**2 - y**2)
    rsharray[2, 2, 1] = np.sqrt(3.0) * x * y
    
    # l=3 (octupole)
    rsharray[3, 0, 0] = 0.5 * (5.0 * z**3 - 3.0 * z * rsq)
    rsharray[3, 1, 0] = 0.25 * np.sqrt(6.0) * (4.0 * x * z**2 - x**3 - x * y**2)
    rsharray[3, 1, 1] = 0.25 * np.sqrt(6.0) * (4.0 * y * z**2 - y * x**2 - y**3)
    rsharray[3, 2, 0] = 0.5 * np.sqrt(15.0) * z * (x**2 - y**2)
    rsharray[3, 2, 1] = np.sqrt(15.0) * x * y * z
    rsharray[3, 3, 0] = 0.25 * np.sqrt(10.0) * (x**3 - 3.0 * x * y**2)
    rsharray[3, 3, 1] = 0.25 * np.sqrt(10.0) * (3.0 * x**2 * y - y**3)
    
    # l=4 (hexadecapole)
    rsharray[4, 0, 0] = 0.125 * (8.0*z**4 - 24.0*(x**2+y**2)*z**2 + 3.0*(x**4+2.0*x**2*y**2+y**4))
    rsharray[4, 1, 0] = 0.25 * np.sqrt(10.0) * (4.0*x*z**3 - 3.0*x*z*(x**2+y**2))
    rsharray[4, 1, 1] = 0.25 * np.sqrt(10.0) * (4.0*y*z**3 - 3.0*y*z*(x**2+y**2))
    rsharray[4, 2, 0] = 0.25 * np.sqrt(5.0) * (x**2-y**2)*(6.0*z**2-x**2-y**2)
    rsharray[4, 2, 1] = 0.25 * np.sqrt(5.0) * x*y*(6.0*z**2-x**2-y**2)
    rsharray[4, 3, 0] = 0.25 * np.sqrt(70.0) * z*(x**3-3.0*x*y**2)
    rsharray[4, 3, 1] = 0.25 * np.sqrt(70.0) * z*(3.0*x**2*y-y**3)
    rsharray[4, 4, 0] = 0.125 * np.sqrt(35.0) * (x**4-6.0*x**2*y**2+y**4)
    rsharray[4, 4, 1] = 0.125 * np.sqrt(35.0) * x*y*(x**2-y**2)
    
    return rsharray[l, m, cs]




# integration bounds 
r1 = 6.78 # inner radius 
r2 = 12.45 # outer radius
small = 1.0e-4 # SVD threshold 
maxl = 4 # maximum multipole order 

inpfile = sys.argv[1] if len(sys.argv) > 1 else "temp_format.dma"
multsites = numbersites(inpfile)
print(f"multsites={multsites}")

multipoles = np.zeros((multsites, maxl+1, maxl+1, 2))
xyzmult    = np.zeros((multsites, 3))

midbond = np.zeros((multsites, multsites), dtype=int)
lmax = np.zeros(multsites, dtype=int)
rvdw = np.zeros(multsites, dtype=np.float64)
atomtype = np.full(multsites, '', dtype='<U2')

midbond[:,:] = 0 # NOTE: the original code references upper triangle matrix, doesn't seem to actually implement it, what's this for? 

# allocate charge array 
# + Count additional charge sites from bonds indicated in midbond (default: none)
count = 0
for i in range(multsites):
    for j in range(i+1, multsites):
        if midbond[i,j] == 1:
            count += 1
chargesites = multsites + count
print(f"chargesites={chargesites}")

xyzcharge = np.zeros((chargesites, 3))
qstore = np.zeros(chargesites)
quse = np.zeros(chargesites) # excluding redundant qstore(:)=0.0

lmax, mm, ms, atomtype = getmultmoments(inpfile, multsites, lmax, multipoles, xyzmult, atomtype)
print(f"lmax={lmax}\nmm={mm}\nms={ms}\natomtype={atomtype}")

qs = gencharges(xyzmult, xyzcharge, midbond)
print(f"qs={qs}")

# Create rvdw, which determines radius encompassing charges
# for each multipole site. For instance, if there is only a monopole
# on hydrogen, make rvdw small so that the monopole is put on hydrogen

# Default initialization
rvdw = np.full(multsites, 3.0)

# Modification for hydrogen
for i in range(multsites):
    # Check if the atom type contains 'H'
    hyd = atomtype[i].find('H')
    
    if hyd == -1:  # No 'H' found
        print(hyd)
        rvdw[i] = 3.0 # NOTE: this essentially takes out certain multipole sites, so is the final partial charge sensitive to this parameter?
    else:
        # Commented out in original code
        # rvdw[i] = 1.0 # NOTE: so the 'H' still end up using rvdw=3.0?
        pass
print(f"rvdw={rvdw}")


# fit charges for each multipole site
# then add them to the total charge array
for i in range(multsites):
    
    # determine charge positions are close enough to fit given multsite
    count = 0
    for j in range(chargesites):
        # calculate distance, r, between charge, q, and multipole, m, sites
        rqm = np.sqrt(
            (xyzmult[i, 0] - xyzcharge[j, 0])**2 +
            (xyzmult[i, 1] - xyzcharge[j, 1])**2 +
            (xyzmult[i, 2] - xyzcharge[j, 2])**2
        )
        print(f"rqm_{i}{j} = {rqm}")
        if rqm < rvdw[i]:
            quse[j] = 1
            count += 1
            print(f"setting quse={quse[j]} for rqm_{i}{j} = {rqm}")
        else:
            quse[j] = 0
    qsites = count
    print(f"qsites={qsites}")

    A = np.zeros((qsites, qsites))
    Astore = np.zeros((qsites, qsites))
    v = np.zeros((qsites, qsites))
    w = np.zeros(qsites)
    q = np.zeros(qsites)
    b = np.zeros(qsites)
    xyzq = np.zeros((qsites, 3))
    btst = np.zeros(qsites)

    # generate xyzq array from xyzcharge array pickout out relevant charges in order from lowest label site 
    count = 0
    for j in range(chargesites):
        if quse[j] == 1:
            xyzq[count, 0] = xyzcharge[j, 0]
            xyzq[count, 1] = xyzcharge[j, 1]
            xyzq[count, 2] = xyzcharge[j, 2]
            count += 1
    print(f"xyzq.shape={xyzq.shape}\nxyzq={xyzq}")

    A = Amat(i, xyzmult, xyzq, r1, r2, lmax[i], A)
    print(f"A.shape={A.shape}\nA={A}")
    b = bvec(i, xyzmult, xyzq, r1, r2, lmax[i], multipoles, b)
    print(f"b.shape={b.shape}\nb={b}")
