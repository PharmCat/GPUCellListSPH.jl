## Details


### Dynamic Particle Collision (DPC) correction.


```math
\\delta \\textbf{v}_i^{DPC} = \\sum k_{ij}\\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \\frac{\\Delta  t}{\\rho_i}\\sum \\phi_{ij} \\frac{2V_j}{V_i + V_j}\\frac{p_{ij}^b}{r_{ij}^2 + \\eta^2}\\textbf{r}_{ij}

\\\\

(v_{ij}^{coll} , \\quad \\phi_{ij}) = \\begin{cases} (\\frac{\\textbf{v}_{ij}\\cdot \\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}\textbf{r}_{ji}, \\quad 0) & \\textbf{v}_{ij}\\cdot \\textbf{r}_{ij} < 0 \\\\ (0, \\quad 1) &  otherwise \\end{cases}

\\\\
p_{ij}^b = \\tilde{p}_{ij} \\chi_{ij} 

\\\\

\\tilde{p}_{ij} = max(min(\\lambda |p_i + p_j|, \\lambda p_{max}), p_{min})

\\\\

\\chi_{ij}  = \\sqrt{\\frac{\\omega({r}_{ij}, l_0)}{\\omega(l_0/2, l_0)}}

\\\\

k_{ij} =  \\begin{cases} \\chi_{ij} & 0.5 \\le {r}_{ij}/l_0 < 1 \\\\ 1 & {r}_{ij}/l_0 < 0.5 \\end{cases}

```

* Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf


