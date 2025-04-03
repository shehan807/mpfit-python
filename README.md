The MPFIT (multipole fitting) python code obtains the partial atomic charge based on the distributed multipole analysis by Stone et al. 

![PyMPFit Logo](PyMPFit_logo.svg)


## Summary 

THE MPFIT code implements Ferenczy's original partial charge fitting procedure to calculate electrostatic charges to reproduce the electrostatic potential of a distributed multipole series. This contrasts with the traditional/popular methods of potential derived charges, i.e., reproducing the molecular electrostatic potential at some atomic sites. In these latter methods, site selection, molecular orientation dependence, or charge assignment of symmetrically related centers seemingly trouble such calculations.

Potential derived charges are, of course, methods in which the charges are not obtained from the wave function, but rather, indirectly via electrostatic potentials. In lieu of computing the entire wave function, constraining some lower moments of the fitted charges to *molecular multipole moments* serves as a better approach to utilizing information inherent to the wave function. In practice, charges based on distributed multipole potentials and electrostatic potentials are equivalent.  
 

## Theory  

From a distributed multipole analysis (DMA), several multipole series at different locations are produced. Charges reproducing the potential of a particle multipole series is the essence of MPFIT. i.e., we seek charges whose multipole moments with respect to the position of the multipole series reproduces the charges obtained from the wave function. 

One begins by defining a function, $f(r)$,

$$f(r) = \sum_{a}\sum_{l,m} Q_{lm}^{a}I_{lm}^{a}(r) - \sum_{i}q_{i}I_{00}^{i}(r)$$

where $f(r)$ refers to a point in space as the difference between the potential of a multipole series at the *a*th site and that of the point charge, $Q_{lm}^{a}$ is a multipole moment (the *m*th component of the rank *l* multipole moment centered at $r_a$), $q_i$ is a point charge in the *i*th position, $I_{lm}^{a}$ is an irregular solid harmonic, $I_{lm}^{a}(r) = r_{a}^{-(l+1)}C_{lm}^{a}(\theta,\phi)$, where $C_{lm}^{a}(\theta,\phi) = [4\pi/(2l + 1)]^{1/2}Y_{lm}^{a}(\theta,\phi)$, and $Y_{lm}^{a}(\theta,\phi)$ is a spherical harmonic. 


Its helpful to see this equation as simply the different in the potential created by the distributed multipole moments, $V^{DMM}(r)$, and the potential created by the point charges, $V^{Q}(r)$, 

$$f(r) = V^{DMM}(r) - V^{Q}(r)$$

Suppose the distance between the multipole center and the point where the potential is to be calculated ($r-r_a$) is greater than the distance between multipoles and the charge site ($r_{ia}$), then 

$$I_{00}^{i}(r) = \sum_{lm} R_{lm}(r_{ia}) I_{lm}^a(r) = \sum_{lm} R_{lm}^a(r_i) I^{a}_{lm}(r)$$

where $R_{lm}(r)$ is a regular spherical harmonic defined as 

$$R_{lm}(r) = r^{2l+1}I_{lm}(r)$$

This allows for simplification of the original equation to, 

<table align="center">
<tr>
<td>

$$f(r) = \sum_a\sum_{l,m}I_{lm}^a(r)[Q_{lm}^a - \sum_{i}q_{i}^{a}R_{lm}^{a}(r_i)] = \sum_{a}f^{a}(r)$$

</td>
</tr>
</table>

We can eliminate $I_{lm}^{a}(r)$ by integration. Namely, the integration of $[f(r)]^2$ yields the optimium set of net charges based on an appropriate integral bound in polar coordinates. This is, in essence, the starting point for the optimization problem:

$$\frac{\delta}{\delta q_{j}^b} \int_{\rho_1}^{\rho_2}\int_{\theta_1}^{\theta_2}\int_{\phi_1}^{\phi_2}[f(r)]^2r^2 \sin{\theta}drd\theta d\phi = 0.$$

Via chain rule, it can be shown that the integrand becomes:

$$\frac{\delta [f(r)]^2}{\delta q_{j}^b} = 2 \sum_a\sum_{l,m}\sum_{l',m'} I_{lm}^{a}(r) I_{l'm'}^{b}(r) R_{l'm'}(r_j) \times \left[\sum_i q_i^a R_{lm}^a(r_i) - Q_{lm}^a\right]$$

which, after plugging back into the integral, can be formulated as 

$$Aq = b$$

where 

$$A_{ij}^{ab} = \sum_{l,m}\sum_{l',m'} K_{lm,l'm'}^{ab} R_{lm}^a (r_i) R_{l'm'}^b (r_j)$$

and 

$$b_j^b = \sum_a\sum_{l,m}\sum_{l',m'} K_{lm,l'm'}^{ab} R_{l'm'}^b(r_j) Q_{lm}^a$$

where 

$$K_{lm,l'm'}^{ab} (\rho_1,\rho_2) = \int_{\rho_1}^{\rho_2}\int_{\theta_1}^{\theta_2}\int_{\phi_1}^{\phi_2} I_{lm}^a(r) I_{l'm'}^b(r) r^2 sin\theta dr d\theta d\phi$$

Computing the off-diagonal components of this approach would be cumbersome ($a\neq b$), especially in the case of having some grid construction involved in molecular electrostatic potential- or field-derived methods. To this end, recall that the optimization can be broken up into the sum of $f(r)$ at each spherical layer around point $a$:

$$F^a(\rho_1,\rho_2) = \sum_{l,m} \frac{4\pi}{2l + 1} W_{\rho_1,\rho_2,l} \left[Q_{lm}^a - \sum_iq_i^aR_{lm}^a(r_i)\right]^2$$

where the $W_{\rho_1,\rho_2,l}$ factors

$$W_{\rho_1,\rho_2,l} = \int_{\rho_1}^{\rho_2} r^2r^{-2(l+1)} dr = \frac{1}{1-2l}\left(\rho_2^{1-2l} - \rho_1^{1-2l}\right)$$

weight the importanc eof the multipoles of rank $l$. Now, we just need to solve for 

<table align="center">
<tr>
<td>

$$\frac{\delta F^a}{\delta q_j^a} = 2 \sum_{l,m} \frac{4\pi}{2l+1} W_{\rho_1,\rho_2,l} \left[ Q_{lm}^a - \sum_i q_iR_{lm}^a(r_i)\right]R_{lm}^a(r_j) = 0$$

</td>
</tr>
</table>

where the new $Aq^a = b$ matrix equation is solved by creating the following $A$ and $b$ matrices:

<table align="center">
<tr>
<td>

$$A_ij^a = \sum_{lm}\frac{1}{2l+1}R_{lm}^a(r_i) R_{l'm'}^b(r_j) W_{\rho_1,\rho_2,l}$$

</td>
</tr>
</table>

<table align="center">
<tr>
<td>

$$b_j^a = \sum_{lm}\frac{1}{2l+1}R_{lm}^a(r_i) Q_{lm}^a W_{\rho_1,\rho_2,l} $$

</td>
</tr>
</table>

where the partial charges are deterimed via 

$$q^a = A^{-1}b$$

where 

<table align="center">
<tr>
<td>

$$q_i = \sum_a q_i^a = \sum_a A_a^{-1}b^a$$

</td>
</tr>
</table>
 
