import sys
import numpy as np 
from scipy.special import sph_harm_y, factorial


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
    for i in range(maxl+1):
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
            for l in range(0, maxl+1):
                if l == 0:
                    _sum = (1.0 / (2.0*l + 1.0)) * W[0] * RSH_scipy(0, 0, 0, xj, yj, zj) * RSH_scipy(0, 0, 0, xk, yk, zk)
                else:
                    for m in range(l+1):
                        if m == 0:
                            print(f"RSH(l={l}, 0, 0, xj={xj}, yj={yj}, zj={zj})={RSH(l, 0, 0, xj, yj, zj)}")
                            print(f"RSH_scipy(l={l}, 0, 0, xj={xj}, yj={yj}, zj={zj})={RSH_scipy(l, 0, 0, xj, yj, zj)}")
                            _sum += (1.0 / (2.0*l + 1.0)) * W[l] * (RSH_scipy(l, 0, 0, xj, yj, zj) * RSH_scipy(l, 0, 0, xk, yk, zk))
                        else:
                            # For m>0, include both real and imaginary parts
                            print(f"RSH(l={l}, m={m}, 0, xj={xj}, yj={yj}, zj={zj})={RSH(l, m, 0, xj, yj, zj)}")
                            print(f"RSH_scipy(l={l}, m={m}, 0, xj={xj}, yj={yj}, zj={zj})={RSH_scipy(l, m, 0, xj, yj, zj)}")
                            _sum += (1.0 / (2.0*l + 1.0)) * W[l] * (
                                RSH_scipy(l, m, 0, xj, yj, zj) * RSH_scipy(l, m, 0, xk, yk, zk) +
                                RSH_scipy(l, m, 1, xj, yj, zj) * RSH_scipy(l, m, 1, xk, yk, zk)
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
                        multipoles[nsite, 0, 0, 0] * RSH_scipy(0, 0, 0, xk, yk, zk)
            else:
                for m in range(l+1):
                    if m == 0: 
                        # m = 0 case
                        _sum += (1.0 / (2.0 * l + 1.0)) * W[l] * \
                                multipoles[nsite, l, 0, 0] * RSH_scipy(l, 0, 0, xk, yk, zk)
                    else:
                        # m > 0 case
                        _sum += (1.0 / (2.0 * l + 1.0)) * W[l] * \
                                   (multipoles[nsite, l, m, 0] * RSH_scipy(l, m, 0, xk, yk, zk) +
                                    multipoles[nsite, l, m, 1] * RSH_scipy(l, m, 1, xk, yk, zk))
        b[k] = _sum
    return b

def RSH_scipy(l, m, cs, x, y, z):
    """Evaluate regular solid harmonics using scipy."""
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1e-16:
        return 1.0 if (l == 0 and m == 0 and cs == 0) else 0.0
    theta = np.arccos(z / r)  
    phi   = np.arctan2(y, x)  
    
    Y = sph_harm_y(l, m, theta, phi)
    
    # 'Normalization' factor to remove from the built-in Y_l^m:
    norm = np.sqrt(4.0 * np.pi / (2.*l + 1.))
    
    if m == 0:
        return norm * r**l * Y.real
    else:
        return np.sqrt(2.) * (-1.)**m * norm * r**l * (Y.real if cs == 0 else Y.imag)

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

def pythag(a, b):
    """
    Compute sqrt(a^2 + b^2) without destructive underflow or overflow.
    This is a faithful translation of the Fortran pythag function.
    """
    absa = abs(a)
    absb = abs(b)
    if absa > absb:
        return absa * np.sqrt(1.0 + (absb/absa)**2)
    else:
        if absb == 0.0:
            return 0.0
        else:
            return absb * np.sqrt(1.0 + (absa/absb)**2)

def svdcmp_sp(a, w, v):
    """
    Singular Value Decomposition (SVD) implementation translated from Fortran.
    This function decomposes matrix 'a' into U*W*V^T where U is stored in 'a',
    W is a diagonal matrix stored in 'w', and V is stored in 'v'.
    """
    # Get dimensions of input matrix
    m = a.shape[0]  # Number of rows
    n = a.shape[1]  # Number of columns
    
    # Check that dimensions match (equivalent to Fortran's assert_eq)
    assert n == v.shape[0] and n == v.shape[1] and n == w.shape[0], 'svdcmp_sp: dimension mismatch'
    
    # Initialize variables (same as Fortran)
    g = 0.0
    scale = 0.0
    
    # Create temporary arrays (equivalent to Fortran's ALLOCATE)
    tempm = np.zeros(m)  # Equivalent to tempm(m) in Fortran
    rv1 = np.zeros(n)  # Equivalent to rv1(n) in Fortran
    tempn = np.zeros(n)  # Equivalent to tempn(n) in Fortran
    
    # Main loop (equivalent to Fortran's DO i=1,n)
    for i in range(n):
        l = i + 1  # Equivalent to l=i+1 in Fortran
        
        # Equivalent to rv1(i)=scale*g in Fortran
        rv1[i] = scale * g
        
        # Reset variables (same as Fortran)
        g = 0.0
        scale = 0.0
        
        # Equivalent to IF (i <= m) THEN in Fortran
        # Note: In Fortran, array indices start at 1, so i <= m means i is less than or equal to m
        # In Python, array indices start at 0, so i < m is the equivalent condition
        if i < m:
            # Equivalent to scale=SUM(ABS(a(i:m,i))) in Fortran
            # SUM in Fortran becomes np.sum in Python
            # ABS in Fortran becomes np.abs in Python
            # a(i:m,i) in Fortran becomes a[i:m,i] in Python
            scale = np.sum(np.abs(a[i:m+1, i]))
            
            # Equivalent to IF (scale /= 0.0) THEN in Fortran
            if scale != 0.0:
                # Equivalent to a(i:m,i)=a(i:m,i)/scale in Fortran
                a[i:m+1, i] = a[i:m+1, i] / scale
                
                # Equivalent to s=DOT_PRODUCT(a(i:m,i),a(i:m,i)) in Fortran
                s = np.dot(a[i:m+1, i], a[i:m+1, i])
                
                # Equivalent to f=a(i,i) in Fortran
                f = a[i, i]
                
                # Equivalent to g=-SIGN(SQRT(s),f) in Fortran
                # SIGN in Fortran becomes np.sign in Python
                # SQRT in Fortran becomes np.sqrt in Python
                g = -np.sign(f) * np.sqrt(s)
                
                # Equivalent to h=f*g-s in Fortran
                h = f * g - s
                
                # Equivalent to a(i,i)=f-g in Fortran
                a[i, i] = f - g
                
                # Equivalent to tempn(l:n)=MATMUL(a(i:m,i),a(i:m,l:n))/h in Fortran
                tempn[l:n+1] = np.dot(a[i:m+1, i], a[i:m+1, l:n+1]) / h
                
                # Equivalent to a(i:m,l:n)=a(i:m,l:n)+outerprod(a(i:m,i),tempn(l:n)) in Fortran
                # outerprod in Fortran becomes np.outer in Python
                a[i:m+1, l:n+1] = a[i:m+1, l:n+1] + np.outer(a[i:m+1, i], tempn[l:n+1])
                
                # Equivalent to a(i:m,i)=scale*a(i:m,i) in Fortran
                a[i:m+1, i] = scale * a[i:m+1, i]
        
        # Equivalent to w(i)=scale*g in Fortran
        w[i] = scale * g
        
        # Reset variables (same as Fortran)
        g = 0.0
        scale = 0.0
        
        # Equivalent to IF ((i <= m) .AND. (i /= n)) THEN in Fortran
        if i < m and i != n-1:  # Note: n-1 because Python is 0-based
            # Equivalent to scale=SUM(ABS(a(i,l:n))) in Fortran
            scale = np.sum(np.abs(a[i, l:n+1]))
            
            # Equivalent to IF (scale /= 0.0) THEN in Fortran
            if scale != 0.0:
                # Equivalent to a(i,l:n)=a(i,l:n)/scale in Fortran
                a[i, l:n+1] = a[i, l:n+1] / scale
                
                # Equivalent to s=DOT_PRODUCT(a(i,l:n),a(i,l:n)) in Fortran
                s = np.dot(a[i, l:n+1], a[i, l:n+1])
                
                # Equivalent to f=a(i,l) in Fortran
                f = a[i, l]
                
                # Equivalent to g=-SIGN(SQRT(s),f) in Fortran
                g = -np.sign(f) * np.sqrt(s)
                
                # Equivalent to h=f*g-s in Fortran
                h = f * g - s
                
                # Equivalent to a(i,l)=f-g in Fortran
                a[i, l] = f - g
                
                # Equivalent to rv1(l:n)=a(i,l:n)/h in Fortran
                rv1[l:n+1] = a[i, l:n+1] / h
                
                # Equivalent to tempm(l:m)=MATMUL(a(l:m,l:n),a(i,l:n)) in Fortran
                tempm[l:m+1] = np.dot(a[l:m+1, l:n+1], a[i, l:n+1])
                
                # Equivalent to a(l:m,l:n)=a(l:m,l:n)+outerprod(tempm(l:m),rv1(l:n)) in Fortran
                a[l:m+1, l:n+1] = a[l:m+1, l:n+1] + np.outer(tempm[l:m+1], rv1[l:n+1])
                
                # Equivalent to a(i,l:n)=scale*a(i,l:n) in Fortran
                a[i, l:n+1] = scale * a[i, l:n+1]
    
    # Equivalent to anorm=MAXVAL(ABS(w)+ABS(rv1)) in Fortran
    # MAXVAL in Fortran becomes np.max in Python
    anorm = np.max(np.abs(w) + np.abs(rv1))
    
    # Equivalent to DO i=n,1,-1 in Fortran
    # Fortran's DO i=n,1,-1 means loop from n down to 1 with step -1
    # In Python, this becomes range(n-1, -1, -1) which gives [n-1, n-2, ..., 0]
    for i in range(n-1, -1, -1):
        
        if i < n-1:  # Note: n-1 because Python is 0-based
            if g != 0.0:
                # Equivalent to v(l:n,i)=(a(i,l:n)/a(i,l))/g in Fortran
                v[l:n+1, i] = (a[i, l:n+1] / a[i, l]) / g
                
                # Equivalent to tempn(l:n)=MATMUL(a(i,l:n),v(l:n,l:n)) in Fortran
                tempn[l:n+1] = np.dot(a[i, l:n+1], v[l:n+1, l:n+1])
                
                # Equivalent to v(l:n,l:n)=v(l:n,l:n)+outerprod(v(l:n,i),tempn(l:n)) in Fortran
                v[l:n+1, l:n+1] = v[l:n+1, l:n+1] + np.outer(v[l:n+1, i], tempn[l:n+1])
            
            # Equivalent to v(i,l:n)=0.0 in Fortran
            v[i, l:n+1] = 0.0
            
            # Equivalent to v(l:n,i)=0.0 in Fortran
            v[l:n+1, i] = 0.0
        
        # Equivalent to v(i,i)=1.0 in Fortran
        v[i, i] = 1.0
        
        # Equivalent to g=rv1(i) in Fortran
        g = rv1[i]
        
        # Equivalent to l=i in Fortran
        l = i
    
    # Equivalent to DO i=MIN(m,n),1,-1 in Fortran
    # MIN in Fortran becomes min in Python
    for i in range(min(m, n)-1, -1, -1):
        l = i + 1
        g = w[i]
        
        # Equivalent to a(i,l:n)=0.0 in Fortran
        a[i, l:n+1] = 0.0
        
        # Equivalent to IF (g /= 0.0) THEN in Fortran
        if g != 0.0:
            # Equivalent to g=1.0_sp/g in Fortran
            # _sp in Fortran indicates single precision
            # In Python, we don't need to specify precision
            g = 1.0 / g
            
            # Equivalent to tempn(l:n)=(MATMUL(a(l:m,i),a(l:m,l:n))/a(i,i))*g in Fortran
            tempn[l:n+1] = (np.dot(a[l:m+1, i], a[l:m+1, l:n+1]) / a[i, i]) * g
            
            # Equivalent to a(i:m,l:n)=a(i:m,l:n)+outerprod(a(i:m,i),tempn(l:n)) in Fortran
            a[i:m+1, l:n+1] = a[i:m+1, l:n] + np.outer(a[i:m+1, i], tempn[l:n+1])
            
            # Equivalent to a(i:m,i)=a(i:m,i)*g in Fortran
            a[i:m+1, i] = a[i:m+1, i] * g
        else:
            # Equivalent to a(i:m,i)=0.0 in Fortran
            a[i:m+1, i] = 0.0
        
        # Equivalent to a(i,i)=a(i,i)+1.0_sp in Fortran
        a[i, i] = a[i, i] + 1.0
    
    # Equivalent to DO k=n,1,-1 in Fortran
    for k in range(n-1, -1, -1):
        # Equivalent to DO its=1,30 in Fortran
        for its in range(30):
            # Equivalent to DO l=k,1,-1 in Fortran
            for l in range(k, -1, -1):
                # Equivalent to nm=l-1 in Fortran
                nm = l - 1
                
                # Equivalent to IF ((ABS(rv1(l))+anorm) == anorm) EXIT in Fortran
                # EXIT in Fortran becomes break in Python
                if abs((np.abs(rv1[l]) + anorm) - anorm) < 1e-14: # == anorm:
                    break
                # Equivalent to IF ((ABS(w(nm))+anorm) == anorm) THEN in Fortran
                if nm >= 0 and abs((np.abs(w[nm]) + anorm) - anorm) < 1e-14: # == anorm:
                    
                    c = 0.0
                    s = 1.0
                    
                    # Equivalent to DO i=l,k in Fortran
                    for i in range(l, k+1):
                        # Equivalent to f=s*rv1(i) in Fortran
                        f = s * rv1[i]
                        
                        # Equivalent to rv1(i)=c*rv1(i) in Fortran
                        rv1[i] = c * rv1[i]
                        
                        # Equivalent to IF ((ABS(f)+anorm) == anorm) EXIT in Fortran
                        if abs((np.abs(f) + anorm) - anorm) < 1e-14: # == anorm:
                            break
                        
                        # Equivalent to g=w(i) in Fortran
                        g = w[i]
                        h = pythag(f, g)
                        w[i] = h
                        
                        # Equivalent to h=1.0_sp/h in Fortran
                        h = 1.0 / h
                        
                        # Equivalent to c= (g*h) in Fortran
                        c = g * h
                        
                        # Equivalent to s=-(f*h) in Fortran
                        s = -(f * h)
                        
                        # Equivalent to tempm(1:m)=a(1:m,nm) in Fortran
                        tempm = a[:, nm].copy()
                        
                        # Equivalent to a(1:m,nm)=a(1:m,nm)*c+a(1:m,i)*s in Fortran
                        a[:, nm] = a[:, nm] * c + a[:, i] * s
                        
                        # Equivalent to a(1:m,i)=-tempm(1:m)*s+a(1:m,i)*c in Fortran
                        a[:, i] = -tempm * s + a[:, i] * c
                    
                    # Equivalent to EXIT in Fortran
                    break
                
            # Equivalent to z=w(k) in Fortran
            z = w[k]
            
            # Equivalent to IF (l == k) THEN in Fortran
            if l == k:
                # Equivalent to IF (z < 0.0) THEN in Fortran
                if z < 0.0:
                    # Equivalent to w(k)=-z in Fortran
                    w[k] = -z
                    
                    # Equivalent to v(1:n,k)=-v(1:n,k) in Fortran
                    v[:, k] = -v[:, k]
                
                # Equivalent to EXIT in Fortran
                break
            
            # Equivalent to IF (its == 30) CALL nrerror('svdcmp_sp: no convergence in svdcmp') in Fortran
            # nrerror in Fortran becomes raise ValueError in Python
            if its == 29:  # 0-based indexing in Python
                raise ValueError('svdcmp_sp: no convergence in svdcmp')
            
            # Equivalent to x=w(l) in Fortran
            x = w[l]
            
            # Equivalent to nm=k-1 in Fortran
            nm = k - 1
            
            # Equivalent to y=w(nm) in Fortran
            y = w[nm]
            
            # Equivalent to g=rv1(nm) in Fortran
            g = rv1[nm]
            
            # Equivalent to h=rv1(k) in Fortran
            h = rv1[k]
            
            # Equivalent to f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0_sp*h*y) in Fortran
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            
            # Equivalent to g=pythag(f,1.0_sp) in Fortran
            g = pythag(f, 1.0)
            
            # Equivalent to f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x in Fortran
            # f = ((x-z)*(x+z)+h*((y/(f+np.sign(g,f)))-h))/x
            f = ((x-z)*(x+z)+h*((y/(f+np.sign(f)*np.abs(g)))-h))/x 
            # Equivalent to c=1.0 in Fortran
            c = 1.0
            
            # Equivalent to s=1.0 in Fortran
            s = 1.0
            # Equivalent to DO j=l,nm in Fortran
            for j in range(l, nm+1):
                # Equivalent to i=j+1 in Fortran
                i = j + 1
                
                # Equivalent to g=rv1(i) in Fortran
                g = rv1[i]
                
                # Equivalent to y=w(i) in Fortran
                y = w[i]
                
                # Equivalent to h=s*g in Fortran
                h = s * g
                
                # Equivalent to g=c*g in Fortran
                g = c * g
                
                # Equivalent to z=pythag(f,h) in Fortran
                z = pythag(f, h)
                
                # Equivalent to rv1(j)=z in Fortran
                rv1[j] = z
                
                # Equivalent to c=f/z in Fortran
                c = f / z
                
                # Equivalent to s=h/z in Fortran
                s = h / z
                
                # Equivalent to f= (x*c)+(g*s) in Fortran
                f = (x*c) + (g*s)
                
                # Equivalent to g=-(x*s)+(g*c) in Fortran
                g = -(x*s) + (g*c)
                
                # Equivalent to h=y*s in Fortran
                h = y * s
                
                # Equivalent to y=y*c in Fortran
                y = y * c
                
                # Equivalent to tempn(1:n)=v(1:n,j) in Fortran
                tempn = v[:, j].copy()
                
                # Equivalent to v(1:n,j)=v(1:n,j)*c+v(1:n,i)*s in Fortran
                v[:, j] = v[:, j] * c + v[:, i] * s
                
                # Equivalent to v(1:n,i)=-tempn(1:n)*s+v(1:n,i)*c in Fortran
                v[:, i] = -tempn * s + v[:, i] * c
                
                # Equivalent to z=pythag(f,h) in Fortran
                z = pythag(f, h)
                
                # Equivalent to w(j)=z in Fortran
                w[j] = z
                
                # Equivalent to IF (z /= 0.0) THEN in Fortran
                if z != 0.0:
                    # Equivalent to z=1.0_sp/z in Fortran
                    z = 1.0 / z
                    
                    # Equivalent to c=f*z in Fortran
                    c = f * z
                    
                    # Equivalent to s=h*z in Fortran
                    s = h * z
                
                # Equivalent to f= (c*g)+(s*y) in Fortran
                f = (c*g) + (s*y)
                
                # Equivalent to x=-(s*g)+(c*y) in Fortran
                x = -(s*g) + (c*y)
                
                # Equivalent to tempm(1:m)=a(1:m,j) in Fortran
                tempm = a[:, j].copy()
                
                # Equivalent to a(1:m,j)=a(1:m,j)*c+a(1:m,i)*s in Fortran
                a[:, j] = a[:, j] * c + a[:, i] * s
                
                # Equivalent to a(1:m,i)=-tempm(1:m)*s+a(1:m,i)*c in Fortran
                a[:, i] = -tempm * s + a[:, i] * c
            
            # Equivalent to rv1(l)=0.0 in Fortran
            rv1[l] = 0.0
            
            # Equivalent to rv1(k)=f in Fortran
            rv1[k] = f
            
            # Equivalent to w(k)=x in Fortran
            w[k] = x
    
    return a, w, v


def svbksb_sp(u, w, v, b):
    """
    Solve A*x = b using the SVD decomposition of A.
    """
    # Get dimensions
    mdum = u.shape[0]
    ndum = u.shape[1]
    
    # Check dimensions match (equivalent to assert_eq in Fortran)
    assert mdum == len(b), 'svbksb_sp: mdum'
    assert ndum == v.shape[0] and ndum == v.shape[1] and ndum == len(w) and ndum == len(b), 'svbksb_sp: ndum'
    
    # Compute U^T * b
    tmp_raw = np.dot(b, u)
    
    # Apply division by w where w is non-zero
    tmp = np.zeros(ndum)
    for j in range(ndum):
        if w[j] != 0.0:
            tmp[j] = tmp_raw[j] / w[j]
    
    # Compute V * tmp
    x = np.dot(v, tmp)

    return x

# integration bounds 
r1 = 6.78 # inner radius 
r2 = 12.45 # outer radius
small = 1.0e-4 # SVD threshold 
maxl = 4 # maximum multipole order 

inpfile = sys.argv[1] if len(sys.argv) > 1 else "gdma/temp_format.dma"
multsites = numbersites(inpfile)

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

xyzcharge = np.zeros((chargesites, 3))
qstore = np.zeros(chargesites)
quse = np.zeros(chargesites) # excluding redundant qstore(:)=0.0

lmax, mm, ms, atomtype = getmultmoments(inpfile, multsites, lmax, multipoles, xyzmult, atomtype)

qs = gencharges(xyzmult, xyzcharge, midbond)

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
        rvdw[i] = 3.0 # NOTE: this essentially takes out certain multipole sites, so is the final partial charge sensitive to this parameter?
    else:
        # Commented out in original code
        # rvdw[i] = 1.0 # NOTE: so the 'H' still end up using rvdw=3.0?
        pass

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
        if rqm < rvdw[i]:
            quse[j] = 1
            count += 1
            #print(f"setting quse={quse[j]} for rqm_{i}{j} = {rqm}")
        else:
            quse[j] = 0
    qsites = count

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

    A = Amat(i, xyzmult, xyzq, r1, r2, lmax[i], A)
    b = bvec(i, xyzmult, xyzq, r1, r2, lmax[i], multipoles, b)

    Astore = A.copy()
    
    A, w, v = svdcmp_sp(A, w, v)

    # Set small singular values to zero (equivalent to lines 119-123 in mpfit.f90)
    for j in range(len(w)):
        if w[j] < small:
            w[j] = 0.0
    
    # Call svbksb_sp to solve the system
    
    q = svbksb_sp(A, w, v, b)
    
    # Test the solution (equivalent to btst=MATMUL(Astore,q) in Fortran)
    btst = np.dot(Astore, q)
    
    # Add the fitted charges to the total array qstore
    count = 0
    for j in range(chargesites):
        if quse[j] == 1:
            qstore[j] = qstore[j] + q[count]
            count += 1

# Print the final charges for each multipole site
for j in range(multsites):
    print(f"{atomtype[j]}: {qstore[j]:8.5f}")
    
    
