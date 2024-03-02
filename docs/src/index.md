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

sing GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools, SPHKernels

path = joinpath(dirname(pathof(GPUCellListSPH)))

fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")

DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = Tuple.(eachrow(DF_POINTS[!, ["Points:0", "Points:2"]])) # Load particles 

dx  = 0.02                  # resolution
h   = 1.2 * sqrt(2) * dx    # smoothinl length
H   = 2h                    # kernel support length
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = H                    # distance for neighborlist
ρ₀  = 1000.0                # reference dencity
m₀  = ρ₀ * dx * dx          # reference mass
α   = 0.01                  # Artificial viscosity constant
g   = 9.81                  # gravity
c₀  = sqrt(g * 2) * 20      # Speed of sound
γ   = 7                     # Gamma costant, used in the pressure equation of state
Δt  = dt  = 1e-5            # time step
δᵩ  = 0.1                   # Coefficient for density diffusion
CFL = 0.2                   # Courant–Friedrichs–Lewy condition for Δt stepping
cellsize = (H, H)           # cell size
sphkernel    = WendlandC2(Float64, 2) # SPH kernel from SPHKernels.jl

system  = GPUCellList(cpupoints, cellsize, dist)
N       = length(cpupoints)
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)
v       = CUDA.fill((0.0, 0.0), length(cpupoints))

sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, v, ptype, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)


timesolve!(sphprob; batch = 10, timeframe = 1.0, writetime = 0.02, path = "D:/vtk/", pvc = true)
```

!!! tip "Save results"
    If you want so save results define `path` and set `writetime`; `pvc = true` make pvd file. To make animation output try `anim = true`.

Other examples available [here](https://github.com/PharmCat/GPUCellListSPH.jl/tree/main/examples).

## Principle algorithm

This part of `stepsolve!` shows how main quations aplied.

```julia
        # kernels for each pair
        W_2d!(prob.W, pairs, x, prob.H⁻¹, prob.sphkernel)
        # kernels gradientfor each pair
        ∇W_2d!(prob.∇W, pairs, x, prob.H⁻¹, prob.sphkernel)
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
    system::GPUCellList
    dim::Int
    dx::T
    h::T                                  # smoothing length
    h⁻¹::T
    H::T                                  # kernel support radius (2h)
    H⁻¹::T
    sphkernel::AbstractSPHKernel          # SPH kernel from SPHKernels.jl
    ∑W::CuArray                           # sum of kernel values
    ∑∇W                                   # sum of kernel gradients
    W::CuArray                            # values of kernel gradient for each pair 
    ∇W::CuArray                           # values of kernel gradient for each pair 
    ∑∂v∂t                                 # acceleration (momentum equation)
    ∑∂ρ∂t                                 # rho diffusion - density derivative function (with diffusion)
    ρ::CuArray                            # rho
    ρΔt½::CuArray                         # rho at t½  
    v::CuArray                            # velocity
    vΔt½::CuArray                         # velocity at t½  
    xΔt½::CuArray                         # coordinates at xΔt½
    P::CuArray                            # pressure (Equation of State in Weakly-Compressible SPH)
    ptype::CuArray                        # particle type: 1 - fluid 1; 0 - boundary; -1 boundary hard layer 
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
    cΔx                                   # cumulative location changes in batch
    nui::T                                # non update interval, update if maximum(maximum.(abs, prob.cΔx)) > 0.9 * prob.nui  
    # Dynamic Particle Collision (DPC) 
    dpc_l₀::T                             # minimal distance
    dpc_pmin::T                           # minimal pressure
    dpc_pmax::T                           # maximum pressure
    dpc_λ::T                              # λ is a non-dimensional adjusting parameter
    # XSPH
    xsph_𝜀::T                             # xsph constant
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
