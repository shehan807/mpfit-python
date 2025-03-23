The MPFIT (multipole fitting) python code obtains the partial atomic charge based on the distributed multipole analysis by Stone et al. 

## Summary 

THE MPFIT code implements Ferenczy's original partial charge fitting procedure to calculate electrostatic charges to reproduce the electrostatic potential of a distributed multipole series. This contrasts with the traditional/popular methods of potential derived charges, i.e., reproducing the molecular electrostatic potential at some atomic sites. In these latter methods, site selection, molecular orientation dependence, or charge assignment of symmetrically related centers seemingly trouble such calculations.

Potential derived charges are, of course, methods in which the charges are not obtained from the wave function, but rather, indirectly via electrostatic potentials. In lieu of computing the entire wave function, constraining some lower moments of the fitted charges to *molecular multipole moments* serves as a better approach to utilizing information inherent to the wave function. In practice, charges based on distributed multipole potentials and electrostatic potentials are equivalent.  
 

## Theory  

From a distributed multipole analysis (DMA), several multipole series at different locations are produced. Charges reproducing the potential of a particle multipole series is the essence of MPFIT. i.e., we seek charges whose multipole moments with respect to the position of the multipole series reproduces the charges obtained from the wave function. 

One begins by defining a function, $f_{ni}$,

$$ f_{ni} = \Sigma_{\lambda\mu} Q_{i\lambda\mu}I_{ni\lambda\mu} - \Sigma_{j}q_{j}I_{nj00}$$

where $n$ refers to a point in space as the difference between the potential of a multipole series at the *i*th site and that of the point charge,$Q_{i\lambda\mu} is a multipole moment (of order $\lambda\mu$ in the *i*th position), $q_j$ is a point charge in the *j*th position, $I_{ni\lambda\mu}$ is an irregular solid harmonic, $I_{ni\lambda\mu} = r_{ni}^{-\lambda - 1}C_{ni\lambda\mu}$, where $C_{ni\lambda\mu} = [4\pi/2\lambda + 1]^{1/2}Y_{ni\lambda\mu}$, and $Y_{ni\lambda\mu}$ is a spherical harmonic. 

% One begins by defining a function, f_{ni}
\begin{align}
f_{ni} = \sigma_{\lambda\mu} Q_{i\lambda\mu}I_{ni\lambda\mu} - \sigma_{j}q_{j}I_{nj00}
\end{align}

\begin{annotate}
\annot{f_{ni}}{The function representing the difference between multipole potential and point charge potential}
\annot{n}{A point in space}
\annot{i}{Index for the site position}
\annot{\sigma_{\lambda\mu}}{Summation over indices $\lambda$ and $\mu$}
\annot{Q_{i\lambda\mu}}{Multipole moment of order $\lambda\mu$ at the $i$th position}
\annot{q_j}{Point charge at the $j$th position}
\annot{I_{ni\lambda\mu}}{Irregular solid harmonic}
\annot{I_{ni\lambda\mu} = r_{ni}^{-\lambda - 1}C_{ni\lambda\mu}}{Definition of irregular solid harmonic}
\annot{C_{ni\lambda\mu} = [4\pi/2\lambda + 1]^{1/2}Y_{ni\lambda\mu}}{Definition of $C_{ni\lambda\mu}$}
\annot{Y_{ni\lambda\mu}}{Spherical harmonic}
\end{annotate}

## Methodology 
