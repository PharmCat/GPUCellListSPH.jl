## Details

`m_i, m_j`

`m_0`



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
\frac{\partial \textbf{v}_i}{\partial t} = - \sum  m_j \left( \frac{b_i}{\rho^2_i} + \frac{b_j}{\rho^2_j} + \Pi_{ij} \right) \nabla_i W_{ij}

```

J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.

### Continuity equation

### Density diffusion term


### XSPH correction

```math
\hat{\textbf{v}_{i}} = - \epsilon \sum m_j \frac{\textbf{v}_{ij}}{\overline{\rho}_{ij}} W_{ij}
```

### Corrected Smoothed Particle Method (CSPM) 


### Dynamic Particle Collision (DPC) correction.


```math
\delta \textbf{v}_i^{DPC} = \sum k_{ij}\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \frac{\Delta  t}{\rho_i}\sum \phi_{ij} \frac{2V_j}{V_i + V_j}\frac{p_{ij}^b}{r_{ij}^2 + \eta^2}\textbf{r}_{ij}

\\

(v_{ij}^{coll} , \quad \phi_{ij}) = \begin{cases} (\frac{\textbf{v}_{ij}\cdot \textbf{r}_{ij}}{r_{ij}^2 + \eta^2}\textbf{r}_{ji}, \quad 0) & \textbf{v}_{ij}\cdot \textbf{r}_{ij} < 0 \\ (0, \quad 1) &  otherwise \end{cases}

\\

p_{ij}^b = \tilde{p}_{ij} \chi_{ij} 

\\

\tilde{p}_{ij} = max(min(\lambda |p_i + p_j|, \lambda p_{max}), p_{min})

\\

\chi_{ij}  = \sqrt{\frac{\omega({r}_{ij}, l_0)}{\omega(l_0/2, l_0)}}

\\

k_{ij} =  \begin{cases} \chi_{ij} & 0.5 \le {r}_{ij}/l_0 < 1 \\ 1 & {r}_{ij}/l_0 < 0.5 \end{cases}

```

* Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf


### Time stepping

