## Details

### Denotation

Particle indexes: ``i``, ``j``;

``W`` - kernel function value;

``\nabla W`` - kernel function gradient;

``h`` -  smoothing length;

``H ~ 2h`` - kernel support length;

``m, m_i, m_j``  - mass;

``m_0`` - reference mass;

``\rho, \rho_i, \rho_j`` - dencity;

``\rho_0`` - reference dencity;

``P, P_i, P_j`` - pressure;

``\Pi`` - viscous energy dissipation term (artificial viscosity);

``\textbf{r}, \textbf{r}_i, \textbf{r}_j`` - particle location (vector);

``\textbf{r}_{ij} = \textbf{r}_{i} - \textbf{r}_{j}`` - particle ``i`` to ``j`` distance (vector);

``r_{ij} = \sqrt{\textbf{r}_{ji} \cdot \textbf{r}_{ji}}`` - particle ``i`` to ``j`` distance;

``\textbf{v}, \textbf{v}_i, \textbf{v}_j`` - velocity (vector);

``\textbf{v}_{ij} = \textbf{v}_{i} - \textbf{v}_{j}`` - relative velocity  (vector);

``V, V_i, V_j`` - volume;

``V_i = \frac{m_i}{\rho_i}; V_j = \frac{m_j}{\rho_j}``

``z_{ij}`` -  vertical distance

### Constants

``g`` - gravity;

``c₀`` - speed of sound at the reference density;

``c_0 = c(\rho_0) = \sqrt{\frac{\partial P}{\partial \rho}}``

``γ = 7`` - gamma costant (pressure equation of state);

``\delta_{\Phi}`` - coefficient for density diffusion;


### Equation of State in Weakly-Compressible SPH

```math
P = c_0^2 \rho_0 * \left[  \left( \frac{\rho}{\rho_0} \right)^{\gamma}  \right]
```

* Monaghan et al., 1999
* Batchelor, 1974

### Artificial Viscosity


```math
\Pi_{ij} = \begin{cases} \frac{- \alpha \overline{c}_{ij} \mu_{ij} + \beta \mu_{ij}^2 }{\overline{\rho}_{ij}} &  \textbf{v}_{ij}\cdot \textbf{r}_{ij} < 0 \\ 0 &  otherwise \end{cases}
```

```math
\overline{c}_{ij}  = \frac{c_i + c_j}{2}
```

```math
\overline{\rho}_{ij} = \frac{\rho_i + \rho_j}{2}

```

Monaghan style artificial viscosity:

```math

\frac{\partial \textbf{v}_i}{\partial t} = - \sum  m_j \Pi_{ij} \nabla_i W_{ij}

```

J. Monaghan, “Smoothed particle hydrodynamics”, Reports on Progress in Physics, 68 (2005), pp. 1703-1759.

### Laminar shear stresse

```math
\frac{\partial \textbf{v}_i}{\partial t} = \sum \frac{m_j}{\rho_j}  \left( 2 \nu_i \frac{\textbf{r}_{ij} \cdot \nabla_i W_{ij} }{r_{ij}^2} \right) \textbf{v}_{ij}
```

### Momentum Equation with Artificial Viscosity

```math
\frac{\partial \textbf{v}_i}{\partial t} = - \sum  m_j \left( \frac{p_i}{\rho^2_i} + \frac{p_j}{\rho^2_j} + \Pi_{ij} \right) \nabla_i W_{ij}

```

J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.

### Continuity equation

### Density diffusion term

```math

\frac{\partial \rho_i}{\partial t} = \sum  m_j \textbf{v}_{ij} \cdot \nabla_i W_{ij} + \delta_{\Phi} h c_0 \sum \Psi_{ij} \cdot \nabla_i W_{ij} \frac{m_j}{\rho_j}
```

```math

\Psi_{ij} = 2 (\rho_{ij}^T + \rho_{ij}^H) \frac{\textbf{r}_{ij}}{r_{ij}^2 + \eta^2}
```

```math
\rho_{ij}^H = \rho_0 \left( \sqrt[\gamma]{\frac{P_{ij}^H + 1}{C_b}} - 1\right)
```

```math
P_{ij}^H = \rho_0 g z_{ij}

```

### XSPH correction


Correction to avoid the particles' disordered movement and prevent penetration between them (Monaghan JJ, 1989).


```math
\hat{\textbf{v}_{i}} = - \epsilon \sum m_j \frac{\textbf{v}_{ij}}{\overline{\rho}_{ij}} W_{ij}
```

### Corrected Smoothed Particle Method (CSPM) 

#### Density Renormalisation.

Corrected Smoothed Particle Method (CSPM) Density Renormalisation (Chen et al. 1999).

```math

\rho_{i}^{norm} = \frac{\sum m_j W_{ij}}{\sum \frac{m_j}{\rho_j} W_{ij}}
```


### Dynamic Particle Collision (DPC) correction.

Dynamic pair-wise Particle Collision (DPC) technique adopting dynamic form of the collision and repulsive terms to improve the pressure field (Jandaghian et al. 2022).

```math

\delta \textbf{v}_i^{DPC} = \sum k_{ij}\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \frac{\Delta  t}{\rho_i}\sum \phi_{ij} \frac{2V_j}{V_i + V_j}\frac{p_{ij}^b}{r_{ij}^2 + \eta^2}\textbf{r}_{ij}
```

```math
(v_{ij}^{coll} , \quad \phi_{ij}) = \begin{cases} (\frac{\textbf{v}_{ij}\cdot \textbf{r}_{ij}}{r_{ij}^2 + \eta^2}\textbf{r}_{ji}, \quad 0) & \textbf{v}_{ij}\cdot \textbf{r}_{ij} < 0 \\ (0, \quad 1) &  otherwise \end{cases}
```

```math
p_{ij}^b = \tilde{p}_{ij} \chi_{ij} 
```

```math
\tilde{p}_{ij} = max(min(\lambda |p_i + p_j|, \lambda p_{max}), p_{min})
```

```math
\chi_{ij}  = \sqrt{\frac{\omega({r}_{ij}, l_0)}{\omega(l_0/2, l_0)}}
```

```math
k_{ij} =  \begin{cases} \chi_{ij} & 0.5 \le {r}_{ij}/l_0 < 1 \\ 1 & {r}_{ij}/l_0 < 0.5 \end{cases}

```

* Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf

### Shifting algorithm

*Not done*

### Time stepping

*TBD*

### Boundary force

The repulsive force exerted by the virtual particle on the fluid particle (Rapaport, 2004), n₁ = 12, n₂ = 4, D = 0.4.

```math
F = D * \frac{\left( (\frac{r_0}{\textbf{r}_{ij}})^{n_1} - (\frac{r_0}{\textbf{r}_{ij}})^{n_2}\right)}{r_{ij}^2}
```

### Reference


* Chen JK, Beraun JE, Carney TC (1999) A corrective smoothed particle method for boundary value problems in heat conduction. Int. J. Num. Meth. Engng. https://doi.org/10.1002/(SICI)1097-0207(19990920)46:2<231::AID-NME672>3.0.CO;2-K

* Carlos Alberto Dutra Fraga Filho, Reflective Boundary Conditions Coupled With the SPH Method for the Three-Dimensional Simulation of Fluid-Structure Interaction With Solid Boundaries, 2023 10.21203/rs.3.rs-3214518/v1 

* Carlos Alberto Dutra Fraga Filho, Julio Tomás Aquije Chacaltana,  Boundary treatment techniques in smoothed particle hydrodynamics: implementations in fluid and thermal sciences and results analysis

* J. J. Monaghan, R. A. Gingold, Shock simulation by the particle method sph, Journal of Computational Physics 52 (1983) 374–389. doi:https: //doi.org/10.1016/0021-9991(83)90036-0

* Monaghan JJ (1989) On the problem of penetration in particle methods. J Comput Phys. https://doi.org/10.1016/0021-9991(89)90032-6

* J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.

* J. Monaghan, “Smoothed particle hydrodynamics”, Reports on Progress in Physics, 68 (2005), pp. 1703-1759.

* M. Jandaghian, A. Krimi, A. R. Zarrati, A. Shakibaeinia, Enhanced weakly-compressible mps method for violent free-surface flows: Role of particle regularization techniques, Journal of Computational Physics 434 (2021) 110202. doi:https://doi.org/10.1016/j.jcp.2021.110202.

* Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf

* J. P. Hughes and D. I. Graham, “Comparison of incompressible and weakly-compressible SPH models for free-surface water flows”, Journal of Hydraulic Research, 48 (2010), pp. 105-117.

* A. Ferrari et al., “A new 3D parallel SPH scheme for free surface flows”, Computers and Fluids, 38 (2009), pp. 1203–1217.

* Gotoh, H., Shibahara, T. and Sakai, T. 2001 “Sub-particle-scale turbulence model for the MPSmethod — Lagrangian flow model for hydraulic engineering,” Comp. Fluid Dyn. J. 9(4)339–347 

* Edmond Y.M. Lo, Songdong Shao, Simulation of near-shore solitary wave mechanics by an incompressible SPH method, 2002


### See also

* Pawan Negi, Prabhu Ramachandran, How to train your solver: Verification of boundary conditions for smoothed particle hydrodynamics, 2022

* https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation

* https://pysph.readthedocs.io/en/1.0a1/reference/equations.html