The MPFIT (multipole fitting) python code obtains the partial atomic charge based on the distributed multipole analysis by Stone et al. 

## Summary 

THE MPFIT code implements Ferenczy's original partial charge fitting procedure to calculate electrostatic charges to reproduce the electrostatic potential of a distributed multipole series. This contrasts with the traditional/popular methods of potential derived charges, i.e., reproducing the molecular electrostatic potential at some atomic sites. In these latter methods, site selection, molecular orientation dependence, or charge assignment of symmetrically related centers seemingly trouble such calculations.

Potential derived charges are, of course, methods in which the charges are not obtained from the wave function, but rather, indirectly via electrostatic potentials. In lieu of computing the entire wave function, constraining some lower moments of the fitted charges to *molecular multipole moments* serves as a better approach to utilizing information inherent to the wave function. In practice, charges based on distributed multipole potentials and electrostatic potentials are equivalent.  
 

## Theory  

From a distributed multipole analysis (DMA), several multipole series at different locations are produced. Charges reproducing the potential of a particle multipole series is the essence of MPFIT. i.e., we seek charges whose multipole moments with respect to the position of the multipole series reproduces the charges obtained from the wave function. 

One begins by defining a function, $f(r)$,

$$f(r) = \sum_{a}\sum_{l,m} Q_{lm}^{a}I_{lm}^{a}(r) - \sum_{i}q_{i}I_{00}^{i}(r)$$

where $f(r)$ refers to a point in space as the difference between the potential of a multipole series at the *a*th site and that of the point charge, $Q_{lm}^{a} is a multipole moment (the *m*th component of the rank *l* multipole moment centered at $r_a$), $q_i$ is a point charge in the *i*th position, $I_{lm}^{a}$ is an irregular solid harmonic, $I_{lm}^{a}(r) = r_{a}^{-(l+1)}C_{lm}^{a}(\theta,\phi)$, where $C_{lm}^{a}(\theta,\phi) = [4\pi/(2l + 1)]^{1/2}Y_{lm}^{a}(\theta,\phi)$, and $Y_{lm}^{a}(\theta,\phi)$ is a spherical harmonic. 


Its helpful to see this equation as simply the different in the potential created by the distributed multipole moments, $V^{DMM}(r)$, and the potential created by the point charges, $V^{Q}(r)$, 

$$f(r) = V^{DMM}(r) - V^{Q}(r)$$

Suppose the distance between multipoles ($r-r_a$) is greater than the distance between multipoles and the charge site ($r_{ia}$), then 

$$I_{00}^{i}(r) = \sum_{lm} R_{lm}(r_{ia}) I_{lm}^a(r) = \sum_{lm} R_{lm}^a(r_i) I^{a}_{lm}(r)$$

where $R_{lm}(r)$ is a regular spherical harmonic defined as 

$$R_{lm}(r) = r^{2l+1}I_{lm}(r)$$

This allows for simplification of the original equation to, 

<div align="center">

| $$f(r) = \sum_a\sum_{l,m}I_{lm}^a(r)[Q_{lm}^a - \sum_{i}q_{i}^{a}R_{lm}^{a}(r_i)] = \sum_{a}f^{a}(r)$$ |

</div>

We can eliminate $I_{lm}^{a}(r)$ by integration. Namely, the integration of $[f(r)]^2$ in yields the optimium set of net charges based on an appropriate integral bound in polar coordinates. This is, in essence, the starting point for the optimization problem:

$$\frac{\delta}{\delta q_{j}^b} \int_{\rho_1}^{\rho_2}\int_{\theta_1}^{\theta_2}\int_{\phi_1}^{\phi_2}[f(r)]^2r^2 \sin{\theta}drd\theta d\phi = 0.$$


