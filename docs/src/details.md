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

``\Pi`` - artificial viscosity term;

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

\\
\\

\overline{c}_{ij}  = \frac{c_i + c_j}{2}

\\
\\

\overline{\rho}_{ij} = \frac{\rho_i + \rho_j}{2}

```

Monaghan style artificial viscosity:

```math

\frac{\partial \textbf{v}_i}{\partial t} = - \sum  m_j \Pi_{ij} \nabla_i W_{ij}

```

J. Monaghan, “Smoothed particle hydrodynamics”, Reports on Progress in Physics, 68 (2005), pp. 1703-1759.


### Momentum Equation with Artificial Viscosity

```math
\frac{\partial \textbf{v}_i}{\partial t} = - \sum  m_j \left( \frac{p_i}{\rho^2_i} + \frac{p_j}{\rho^2_j} + \Pi_{ij} \right) \nabla_i W_{ij}

```

J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.

### Continuity equation

### Density diffusion term

```math

\frac{\partial \rho_i}{\partial t} = \sum  m_j \textbf{v}_{ij} \cdot \nabla_i W_{ij} + \delta_{\Phi} h c_0 \sum \Psi_{ij} \cdot \nabla_i W_{ij} \frac{m_j}{\rho_j}


\\
\\


\Psi_{ij} = 2 (\rho_{ij}^T + \rho_{ij}^H) \frac{\textbf{r}_{ij}}{r_{ij}^2 + \eta^2}

\\

\rho_{ij}^H = \rho_0 \left( \sqrt[\gamma]{\frac{P_{ij}^H + 1}{C_b}} - 1\right)

\\

P_{ij}^H = \rho_0 g z_{ij}

```


### XSPH correction

```math
\hat{\textbf{v}_{i}} = - \epsilon \sum m_j \frac{\textbf{v}_{ij}}{\overline{\rho}_{ij}} W_{ij}
```

### Corrected Smoothed Particle Method (CSPM) 


### Dynamic Particle Collision (DPC) correction.


```math

\delta \textbf{v}_i^{DPC} = \sum k_{ij}\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \frac{\Delta  t}{\rho_i}\sum \phi_{ij} \frac{2V_j}{V_i + V_j}\frac{p_{ij}^b}{r_{ij}^2 + \eta^2}\textbf{r}_{ij}

\\
\\

(v_{ij}^{coll} , \quad \phi_{ij}) = \begin{cases} (\frac{\textbf{v}_{ij}\cdot \textbf{r}_{ij}}{r_{ij}^2 + \eta^2}\textbf{r}_{ji}, \quad 0) & \textbf{v}_{ij}\cdot \textbf{r}_{ij} < 0 \\ (0, \quad 1) &  otherwise \end{cases}

\\
\\

p_{ij}^b = \tilde{p}_{ij} \chi_{ij} 

\\
\\

\tilde{p}_{ij} = max(min(\lambda |p_i + p_j|, \lambda p_{max}), p_{min})

\\
\\

\chi_{ij}  = \sqrt{\frac{\omega({r}_{ij}, l_0)}{\omega(l_0/2, l_0)}}

\\
\\

k_{ij} =  \begin{cases} \chi_{ij} & 0.5 \le {r}_{ij}/l_0 < 1 \\ 1 & {r}_{ij}/l_0 < 0.5 \end{cases}

```

* Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf

### Shifting algorithm


### Time stepping

