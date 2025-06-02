import sys
import numpy as np
from scipy.special import sph_harm_y
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

def print_matrix_rich(matrix, title="Matrix", precision=6, max_rows=20, max_cols=10):
    """
    Print a matrix using rich library with beautiful formatting
    
    Parameters:
    -----------
    matrix : np.ndarray
        The matrix to print
    title : str
        Title for the matrix
    precision : int
        Number of decimal places to show
    max_rows : int
        Maximum number of rows to display
    max_cols : int
        Maximum number of columns to display
    """
    console = Console()
    
    # Handle 1D arrays (vectors)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    
    rows, cols = matrix.shape
    
    # Create table
    table = Table(title=f"{title} ({rows}×{cols})", box=box.ROUNDED)
    
    # Add column headers
    table.add_column("Row", style="cyan", no_wrap=True)
    display_cols = min(cols, max_cols)
    for j in range(display_cols):
        table.add_column(f"Col {j}", style="magenta", justify="right")
    
    if cols > max_cols:
        table.add_column("...", style="dim")
    
    # Add rows
    display_rows = min(rows, max_rows)
    for i in range(display_rows):
        row_data = [f"[cyan]{i}[/cyan]"]
        for j in range(display_cols):
            value = matrix[i, j]
            if abs(value) < 1e-10:
                row_data.append("[dim]0.000000[/dim]")
            else:
                row_data.append(f"{value:.{precision}f}")
        
        if cols > max_cols:
            row_data.append("[dim]...[/dim]")
        
        table.add_row(*row_data)
    
    if rows > max_rows:
        ellipsis_row = ["[dim]...[/dim]"] * (display_cols + 1 + (1 if cols > max_cols else 0))
        table.add_row(*ellipsis_row)
    
    console.print(table)
    console.print()

def print_vector_rich(vector, title="Vector", precision=6, max_elements=15):
    """
    Print a vector in a compact, readable format
    """
    console = Console()
    
    vector = np.asarray(vector).flatten()
    n = len(vector)
    
    # Create a compact representation
    if n <= max_elements:
        elements = [f"{v:.{precision}f}" for v in vector]
        vector_str = " ".join(f"[magenta]{elem}[/magenta]" for elem in elements)
    else:
        # Show first few and last few elements
        show_each = max_elements // 2
        first_elements = [f"{v:.{precision}f}" for v in vector[:show_each]]
        last_elements = [f"{v:.{precision}f}" for v in vector[-show_each:]]
        
        first_str = " ".join(f"[magenta]{elem}[/magenta]" for elem in first_elements)
        last_str = " ".join(f"[magenta]{elem}[/magenta]" for elem in last_elements)
        vector_str = f"{first_str} [dim]...[/dim] {last_str}"
    
    panel = Panel(
        vector_str,
        title=f"[bold cyan]{title}[/bold cyan] (length: {n})",
        border_style="blue"
    )
    console.print(panel)
    console.print()

def print_svd_components(U, S, Vh, precision=6):
    """
    Print SVD components in a organized way
    """
    console = Console()
    
    console.print("[bold green]SVD Decomposition: A = U × S × Vᵀ[/bold green]")
    console.print()
    
    print_matrix_rich(U, "U (Left singular vectors)", precision)
    print_vector_rich(S, "S (Singular values)", precision)
    print_matrix_rich(Vh, "Vᵀ (Right singular vectors)", precision)

# Example usage function for your specific matrices
def debug_linear_system(A, b, U, S, Vh, q, site_index, precision=6):
    """
    Print all matrices for debugging the linear system at a specific site
    """
    console = Console()
    
    # Header for this site
    header = Text(f"DEBUGGING SITE {site_index}", style="bold yellow on blue")
    console.print(Panel(header, expand=False))
    console.print()
    
    # Print the main system components
    print_matrix_rich(A, f"System Matrix A", precision)
    print_vector_rich(b, f"Right-hand side b", precision)
    
    # Print SVD components
    print_svd_components(U, S, Vh, precision)
    
    # Print solution
    print_vector_rich(q, f"Solution vector q", precision)
    
    # Print some diagnostics
    console.print(f"[yellow]Matrix condition number:[/yellow] {np.linalg.cond(A):.2e}")
    console.print(f"[yellow]Residual norm ||Aq - b||:[/yellow] {np.linalg.norm(A @ q - b):.2e}")
    console.print()
    console.print("-" * 80)
    console.print()

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
        q_norm_squared = mm[i, l, 0, 0] ** 2
        for j in range(1, l + 1):
            q_norm_squared += mm[i, l, j, 0] ** 2 + mm[i, l, j, 1] ** 2
        q_norm = np.sqrt(q_norm_squared)

        print(f"|Q{l}| = {q_norm:10.6f}  Q{l}0  = {mm[i, l, 0, 0]:10.6f}", end="")

        # Print components
        for j in range(1, l + 1):
            if j == 1:
                print(
                    f"  Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}",
                    end="",
                )
            else:
                # For j > 1, print on new line with spacing
                if j == 2:
                    print()
                print(
                    f"                   Q{l}{j}c = {mm[i, l, j, 0]:10.6f}  Q{l}{j}s = {mm[i, l, j, 1]:10.6f}",
                    end="",
                )
        print()


def numbersites(inpfile):
    count = 0
    with open(inpfile, "r") as f:
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
    mm,  # multipole moments
    ms,  # multipole sites
    atomtype,
    reprint_mm=False,
):
    with open(inpfile, "r") as f:
        for i in range(n):
            line = f.readline().split()
            atomtype[i] = line[0]
            x, y, z = float(line[1]), float(line[2]), float(line[3])
            lmax[i] = int(line[4])

            ms[i, 0] = x
            ms[i, 1] = y
            ms[i, 2] = z

            q0 = float(f.readline().strip())  # monopole
            mm[i, 0, 0, 0] = q0

            if lmax[i] > 0:
                for l in range(1, lmax[i] + 1):
                    line = f.readline().split()
                    mm[i, l, 0, 0] = float(line[0])  # Q_l0
                    for m in range(1, l + 1):  # Q_lm (m>0)
                        idx = 2 * m - 1
                        mm[i, l, m, 0] = float(line[idx])  # real
                        mm[i, l, m, 1] = float(line[idx + 1])  # imaginary

        if reprint_mm:
            # After the with open block:
            for i in range(n):
                print(f"Site {i + 1}:")
                _print_multipole_moments(i, mm, lmax)
                print()
    return lmax, mm, ms, atomtype


def gencharges(ms, qs, midbond, pyrazine_vsites=True):
    """Generate charge positions from multipole sites and bond information"""
    nmult = ms.shape[0]  # number of multipole sites
    nmid = qs.shape[0] - nmult  # number of midpoints

    # copy multipole site coordinates to charge sites
    for i in range(nmult):
        qs[i, 0] = ms[i, 0]
        qs[i, 1] = ms[i, 1]
        qs[i, 2] = ms[i, 2]
    
    # pyrazine special case
    if pyrazine_vsites:
        qs[i+1, 0] = 0.0 
        qs[i+1, 1] = 0.0
        qs[i+1, 2] = 0.0

    if nmid > 0:
        count = 0
        for i in range(nmult):
            for j in range(i + 1, nmult):
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
    ncharge = xyzcharge.shape[0]  # or len(xyzcharge)

    W = np.zeros(maxl + 1)
    # compute W integration factor
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))

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
            for l in range(0, maxl + 1):
                if l == 0:
                    _sum = (
                        (1.0 / (2.0 * l + 1.0))
                        * W[0]
                        * RSH(0, 0, 0, xj, yj, zj)
                        * RSH(0, 0, 0, xk, yk, zk)
                    )
                else:
                    for m in range(l + 1):
                        if m == 0:
                            _sum += (
                                (1.0 / (2.0 * l + 1.0))
                                * W[l]
                                * (RSH(l, 0, 0, xj, yj, zj) * RSH(l, 0, 0, xk, yk, zk))
                            )
                        else:
                            # For m>0, include both real and imaginary parts
                            _sum += (
                                (1.0 / (2.0 * l + 1.0))
                                * W[l]
                                * (
                                    RSH(l, m, 0, xj, yj, zj) * RSH(l, m, 0, xk, yk, zk)
                                    + RSH(l, m, 1, xj, yj, zj)
                                    * RSH(l, m, 1, xk, yk, zk)
                                )
                            )
            A[j, k] = _sum
    return A


def bvec(nsite, xyzmult, xyzcharge, r1, r2, maxl, multipoles, b):
    """Construct b vector as in  J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)"""
    ncharge = xyzcharge.shape[0]

    W = np.zeros(maxl + 1, dtype=np.float64)
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))
    for k in range(ncharge):
        # Compute relative coordinates
        xk = xyzcharge[k, 0] - xyzmult[nsite, 0]
        yk = xyzcharge[k, 1] - xyzmult[nsite, 1]
        zk = xyzcharge[k, 2] - xyzmult[nsite, 2]

        _sum = 0.0
        for l in range(maxl + 1):
            if l == 0:
                # Special case for l = 0
                _sum = (
                    (1.0 / (2.0 * l + 1.0))
                    * W[0]
                    * multipoles[nsite, 0, 0, 0]
                    * RSH(0, 0, 0, xk, yk, zk)
                )
            else:
                for m in range(l + 1):
                    if m == 0:
                        # m = 0 case
                        _sum += (
                            (1.0 / (2.0 * l + 1.0))
                            * W[l]
                            * multipoles[nsite, l, 0, 0]
                            * RSH(l, 0, 0, xk, yk, zk)
                        )
                    else:
                        # m > 0 case
                        _sum += (
                            (1.0 / (2.0 * l + 1.0))
                            * W[l]
                            * (
                                multipoles[nsite, l, m, 0] * RSH(l, m, 0, xk, yk, zk)
                                + multipoles[nsite, l, m, 1] * RSH(l, m, 1, xk, yk, zk)
                            )
                        )
        b[k] = _sum
    return b


def RSH(l, m, cs, x, y, z):
    """Evaluate regular solid harmonics using scipy."""
    r = np.sqrt(x * x + y * y + z * z)
    if r < 1e-10:
        return 1.0 if (l == 0 and m == 0 and cs == 0) else 0.0
    if l == 4:
        # Initialize array for RSH values
        rsharray = np.zeros((5, 5, 2))
        rsq = x**2 + y**2 + z**2    
        
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
    
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    Y = sph_harm_y(l, m, theta, phi)

    # 'Normalization' factor to remove from the built-in Y_l^m:
    norm = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))

    if m == 0:
        return norm * r**l * Y.real
    else:
        return (
            np.sqrt(2.0) * (-1.0) ** m * norm * r**l * (Y.real if cs == 0 else Y.imag)
        )


# integration bounds
r1 = 6.78  # inner radius
r2 = 12.45  # outer radius
small = 1.0e-4  # SVD threshold
maxl = 4  # maximum multipole order

# virtual sites
pyrazine_vsites = True

inpfile = sys.argv[1] if len(sys.argv) > 1 else "gdma/temp_format.dma"
multsites = numbersites(inpfile)

multipoles = np.zeros((multsites, maxl + 1, maxl + 1, 2))
xyzmult = np.zeros((multsites, 3))

midbond = np.zeros((multsites, multsites), dtype=int)
lmax = np.zeros(multsites, dtype=int)
rvdw = np.zeros(multsites, dtype=np.float64)
atomtype = np.full(multsites, "", dtype="<U2")

midbond[:, :] = (
    0  # NOTE: the original code references upper triangle matrix, doesn't seem to actually implement it, what's this for?
)

# allocate charge array
# + Count additional charge sites from bonds indicated in midbond (default: none)
count = 0
for i in range(multsites):
    for j in range(i + 1, multsites):
        if midbond[i, j] == 1:
            count += 1
if pyrazine_vsites:
    count += 1

chargesites = multsites + count

xyzcharge = np.zeros((chargesites, 3))
qstore = np.zeros(chargesites)
quse = np.zeros(chargesites)  # excluding redundant qstore(:)=0.0

lmax, mm, ms, atomtype = getmultmoments(
    inpfile, multsites, lmax, multipoles, xyzmult, atomtype
)

qs = gencharges(xyzmult, xyzcharge, midbond, pyrazine_vsites=pyrazine_vsites)

# Create rvdw, which determines radius encompassing charges
# for each multipole site. For instance, if there is only a monopole
# on hydrogen, make rvdw small so that the monopole is put on hydrogen

# Default initialization
rvdw = np.full(multsites, 3.0)

# Modification for hydrogen
for i in range(multsites):
    # Check if the atom type contains 'H'
    hyd = atomtype[i].find("H")

    if hyd == -1:  # No 'H' found
        rvdw[i] = (
            3.0  # NOTE: this essentially takes out certain multipole sites, so is the final partial charge sensitive to this parameter?
        )
    else:
        # Commented out in original code
        # rvdw[i] = 1.0 # NOTE: so the 'H' still end up using rvdw=3.0?
        pass

# fit charges for each multipole site
# then add them to the total charge array
for i in range(multsites):
    print(f"xyzmult[i]: {xyzmult[i]}")
    print(f"xyzcharge: {xyzcharge}")
    rqm = np.linalg.norm(xyzmult[i] - xyzcharge, axis=1)
    quse_mask = rqm < rvdw[i]

    qsites = np.count_nonzero(quse_mask)

    A = np.zeros((qsites, qsites))
    q = np.zeros(qsites)
    b = np.zeros(qsites)

    xyzq = xyzcharge[quse_mask]
    A = Amat(i, xyzmult, xyzq, r1, r2, lmax[i], A)
    b = bvec(i, xyzmult, xyzq, r1, r2, lmax[i], multipoles, b)

    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    S[S < small] = 0.0

    inv_S = np.zeros_like(S)
    mask_S = S != 0
    inv_S[mask_S] = 1.0 / S[mask_S]

    # shape handling for matrix multiply:
    q = (Vh.T * inv_S) @ (U.T @ b)

    # Add the fitted charges to the total array qstore
    qstore[quse_mask] += q

    debug_linear_system(A, b, U, S, Vh, q, i, precision=6)
    print(quse_mask)
    print(qstore)
# Print the final charges for each multipole site
for j in range(multsites):
    print(f"{atomtype[j]}: {qstore[j]:8.5f}")
if pyrazine_vsites:
    j += 1
    print(f"vsite: {qstore[j]:8.5f}")
print(f"Sum charges: {sum(qstore)}.")
