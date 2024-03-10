```@meta
CurrentModule = GPUCellListSPH
```
# GPUCellListSPH

## Introduction 

Smoothed-particle hydrodynamics (SPH) is a computational method used for simulating the mechanics of continuum media, such as solid mechanics and fluid flows. It was developed by Gingold and Monaghan and Lucy in 1977, initially for astrophysical problems. It has been used in many fields of research, including astrophysics, ballistics, volcanology, and oceanography. It is a meshfree Lagrangian method (where the co-ordinates move with the fluid), and the resolution of the method can easily be adjusted with respect to variables such as density. 

Initially package was based on [AhmedSalih3d](https://github.com/AhmedSalih3d) work ([SPHExample](https://github.com/AhmedSalih3d/SPHExample)) and then taken new features.

See also [Smoothed Particle Hydrodynamics Manual](https://ahmedsalih3d.github.io/SmoothedParticleHydrodynamicsManual/).

Package working with CUDA.jl on GPU and not applicable for lunch on CPU. 

!!! note "Main goals"
    Main goal of this package - make clear and readable implementation of SPH. Performance wasn't primary goal of this package so desing and used algorithms for computation on GPU and memory organization coud be not so efficient.  

## First step

Simple example to try SPH:

```julia

using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK

path = joinpath(dirname(pathof(GPUCellListSPH)))
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")
DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2"]])...)
#cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2"]]))...)

dx  = 0.02                  # resolution
h   = 1.2 * sqrt(2) * dx    # smoothinl length
H   = 2h                    # kernel support length
dist = 1.1H                 # distance for neighborlist
ρ₀  = 1000.0                # Reference density
m₀  = ρ₀ * dx * dx          # Reference mass
α   = 0.01                  # Artificial viscosity constant
g   = 9.81                  # gravity
c₀  = sqrt(g * 2) * 20      # Speed of sound
γ   = 7                     # Gamma costant, used in the pressure equation of state
Δt  = dt  = 1e-5            # Delta time
δᵩ  = 0.1                   # Coefficient for density diffusion
CFL = 0.2                   # Courant–Friedrichs–Lewy condition for Δt stepping
cellsize = (dist, dist)     # cell size
sphkernel    = WendlandC2(Float64, 2) # SPH kernel from SPHKernels.jl

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)

ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)


sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, ptype, ρ₀, m₀, Δt, α,  c₀, γ, δᵩ, CFL; s = 0.0)

# batch - number of iteration until check time and vtp
# timeframe - simulation time
# writetime - write vtp file each interval
# path - path to vtp files
# pvc - make paraview collection
timesolve!(sphprob; batch = 100, timeframe = 1.0, writetime = 0.0, path = "D:/vtk/", pvc = true)
```

!!! tip "Save results"
    If you want so save results define `path` and set `writetime`; `pvc = true` make pvd file. To make animation output try `anim = true`.

Other examples available [here](https://github.com/PharmCat/GPUCellListSPH.jl/tree/main/examples).

## Principle algorithm

This part of `stepsolve!` shows how main quations aplied.

```julia
        # v 1.0.1-DEV
        # kernels for each pair
        sphW!(prob.W, pairs, x, prob.H⁻¹, prob.sphkernel)
        # kernels gradientfor each pair
        sph∇W!(prob.∇W, pairs, x, prob.H⁻¹, prob.sphkernel)
        # density derivative with density diffusion
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, pairs, prob.∇W, prob.ρ, prob.v, x, prob.h, prob.m₀, prob.ρ₀, prob.c₀, prob.γ, prob.g, prob.δᵩ, prob.ptype; minthreads = 256) 
        #  pressure
        pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇W, prob.P, pairs,  prob.m₀, prob.ρ, prob.ptype) 
        # add artificial viscosity
        ∂v∂t_av!(prob.∑∂v∂t, prob.∇W, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀, prob.ptype)
        # laminar shear stresse
        if prob.𝜈 > 0
            ∂v∂t_visc!(prob.∑∂v∂t, prob.∇W, prob.v, prob.ρ, x, pairs, prob.h, prob.m₀, prob.𝜈, prob.ptype)
        end
        # add gravity 
        ∂v∂t_addgrav!(prob.∑∂v∂t, gravvec(prob.g, prob.dim)) 
        #  Boundary forces
        fbmolforce!(prob.∑∂v∂t, pairs, x, 0.4, 2 * prob.dx, prob.ptype)
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, x, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        # following steps (update_ρ!, update_vp∂v∂tΔt!, update_xpvΔt!) can be done in one kernel 
        # calc ρ at Δt½
        update_ρp∂ρ∂tΔt!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.ptype)
        # calc v at Δt½
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ptype) 
        # calc x at Δt½
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5)
        # set derivative to zero for Δt½ calc
        for vec in prob.∑∂v∂t fill!(vec, zero(T)) end
        # density derivative with density diffusion at  xΔt½ 
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, pairs, prob.∇W, prob.ρΔt½, prob.vΔt½, prob.xΔt½, prob.h, prob.m₀, prob.ρ₀, prob.c₀, prob.γ, prob.g, prob.δᵩ, prob.ptype; minthreads = 256) 
        #  pressure
        pressure!(prob.P, prob.ρΔt½, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t, prob.∇W, prob.P, pairs,  prob.m₀, prob.ρΔt½, prob.ptype)
        # add artificial viscosity at xΔt½ 
        ∂v∂t_av!(prob.∑∂v∂t, prob.∇W, pairs, prob.xΔt½, prob.h, prob.ρΔt½, prob.α, prob.vΔt½, prob.c₀, prob.m₀, prob.ptype)
        # laminar shear stresse
        if prob.𝜈 > 0
            ∂v∂t_visc!(prob.∑∂v∂t, prob.∇W, prob.vΔt½, prob.ρΔt½, prob.xΔt½, pairs, prob.h, prob.m₀, prob.𝜈, prob.ptype)
        end
        # add gravity 
        ∂v∂t_addgrav!(prob.∑∂v∂t,gravvec(prob.g, prob.dim))
        #  Boundary forces
        fbmolforce!(prob.∑∂v∂t, pairs, x, prob.bound_D, prob.bound_l, prob.ptype)
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, prob.xΔt½, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        # update all with symplectic position Verlet scheme
        symplectic_update!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.cΔx, prob.ρ₀, prob.ptype)
        # Dynamic Particle Collision (DPC) 
        if prob.dpc_l₀ > 0
            #  pressure
            pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
            dpcreg!(prob.buf2, prob.v, prob.ρ, prob.P, pairs, x, prob.sphkernel, prob.dpc_l₀, prob.dpc_pmin, prob.dpc_pmax, prob.Δt, prob.dpc_λ, dpckernlim, prob.ptype)  
            update_dpcreg!(prob.v, x, prob.buf2, prob.Δt, prob.ptype)
        end
        # XSPH correction.
        if prob.xsph_𝜀 > 0
            xsphcorr!(prob.buf2, pairs, prob.W, prob.ρ, prob.v, prob.m₀, prob.xsph_𝜀, prob.ptype)
            update_xsphcorr!(prob.v, prob.buf2, prob.ptype) 
        end
        # Density Renormalisation every 15 timesteps
        if prob.cspmn > 0 && cspmcorrn == prob.cspmn
            cspmcorr!(prob.buf2, prob.W, prob.ρ, prob.m₀, pairs, prob.ptype)
            cspmcorrn = 0
        end
```

## Docs

### Cell list

```@docs
GPUCellListSPH.GPUCellList
```

```@docs
GPUCellListSPH.update!
```

```@docs
GPUCellListSPH.partialupdate!
```

```@docs
GPUCellListSPH.neighborlist
```

### SPH problem object

```@docs
GPUCellListSPH.SPHProblem
```

Object structure:

``` 
    # v 1.0.1-DEV
    system::GPUCellList                         # Neigbor list system
    dim::Int
    dx::T
    h::T                                        # smoothing length
    h⁻¹::T
    H::T                                        # kernel support radius (2h)
    H⁻¹::T
    sphkernel::AbstractSPHKernel                # SPH kernel from SPHKernels.jl
    ∑W::CuArray                                 # sum of kernel values
    ∑∇W                                         # sum of kernel gradients
    W::CuArray                                  # values of kernel gradient for each pair 
    ∇W                                          # values of kernel gradient for each pair 
    ∑∂v∂t                                       # acceleration (momentum equation)
    ∑∂ρ∂t                                       # rho diffusion - density derivative function (with diffusion)
    ρ::CuArray                                  # rho
    ρΔt½::CuArray                               # rho at t½  
    v                                           # velocity
    vΔt½                                        # velocity at t½  
    xΔt½                                        # coordinates at xΔt½
    P::CuArray                                  # pressure (Equation of State in Weakly-Compressible SPH)
    ptype::CuArray                              # particle type: 1 - fluid 1; 0 - boundary; -1 boundary hard layer 
    ρ₀::T                                 # Reference density
    m₀::T                                 # Initial mass
    Δt::T                                 # default Δt
    α::T                                  # Artificial viscosity alpha constant
    𝜈::T                                  # kinematic fluid viscosity
    g::T                                  # gravity constant
    c₀::T                                 # speed of sound
    γ                                     # Gamma, 7 for water (used in the pressure equation of state)
    s::T                                  # surface tension constant
    δᵩ::T                                 # Coefficient for density diffusion, typically 0.1
    CFL::T                                # CFL number for the simulation 
    buf::CuArray                          # buffer for dt calculation
    buf2                                  # buffer 
    etime::T                              # simulation time
    # For neigbors update
    cΔx                                   # cumulative location changes in batch
    nui::T                                # non update interval, update if maximum(maximum.(abs, prob.cΔx)) > 0.9 * prob.nui  
    # Dynamic Particle Collision (DPC) 
    dpc_l₀::T       # minimal distance
    dpc_pmin::T     # minimal pressure
    dpc_pmax::T     # maximum pressure
    dpc_λ::T        # λ is a non-dimensional adjusting parameter
    # XSPH
    xsph_𝜀::T       # xsph constant
    # CSPM
    cspmn::Int      # step for CSPM (in batch)
    # Bound force
    bound_D::T      # D constant for bounr repulsive force
    bound_l::T      # length for bounr repulsive force (> sqrt(dim-1)dx)
```

### Processing functions

```@docs
GPUCellListSPH.stepsolve!
```

```@docs
GPUCellListSPH.timesolve!
```


### Main equations 

```@docs
GPUCellListSPH.∑W_2d!
```

```@docs
GPUCellListSPH.∑∇W_2d!
```

```@docs
GPUCellListSPH.∂ρ∂tDDT!
```

```@docs
GPUCellListSPH.pressure!
```

```@docs
GPUCellListSPH.∂v∂t!
```

```@docs
GPUCellListSPH.∂v∂t_av!
```

```@docs
GPUCellListSPH.∂v∂t_visc!
```

```@docs
GPUCellListSPH.∂v∂t_addgrav!
```


```@docs
GPUCellListSPH.∂v∂tpF!
```

```@docs
GPUCellListSPH.dpcreg!
```

```@docs
GPUCellListSPH.cspmcorr!
```

```@docs
GPUCellListSPH.xsphcorr!
```

```@docs
GPUCellListSPH.fbmolforce!
```

### Export functions 


```@docs
GPUCellListSPH.makedf
```

```@docs
GPUCellListSPH.writecsv
```

## Reference

 * R.A. Gingold; J.J. Monaghan (1977). "Smoothed particle hydrodynamics: theory and application to non-spherical stars". Mon. Not. R. Astron. Soc. 181 (3): 375–89. Bibcode:1977MNRAS.181..375G. doi:10.1093/mnras/181.3.375.

* L.B. Lucy (1977). "A numerical approach to the testing of the fission hypothesis". Astron. J. 82: 1013–1024. Bibcode:1977AJ.....82.1013L. doi:10.1086/112164.
