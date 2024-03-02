# GPUCellListSPH.jl

## Description

Smoothed-particle hydrodynamics (SPH) is a computational method used for simulating the mechanics of continuum media, such as solid mechanics and fluid flows. It was developed by Gingold and Monaghan and Lucy in 1977, initially for astrophysical problems. It has been used in many fields of research, including astrophysics, ballistics, volcanology, and oceanography. It is a meshfree Lagrangian method (where the co-ordinates move with the fluid), and the resolution of the method can easily be adjusted with respect to variables such as density. 

This is concept-project for particle cell neiborhood list and SPH on GPU (with CUDA.jl). Based on [AhmedSalih3d](https://github.com/AhmedSalih3d) repo [SPHExample](https://github.com/AhmedSalih3d/SPHExample) with ome new features.

## Documentation

[![Latest docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pharmcat.github.io/GPUCellListSPH.jl/dev/)

### Video 

* SPH simulation for dam-break problem - with kinematic viscosity

[![SPH simulation for dam-break problem - with kinematic viscosity](https://www.youtube.com/watch?v=ogktG7Hm8eQ/default.jpg)](https://www.youtube.com/watch?v=ogktG7Hm8eQ)



* SPH simulation for dam-break problem (DPC)

[![SPH simulation for dam-break problem (DPC)](https://www.youtube.com/watch?v=IsE6g05ioVI/default.jpg)](https://www.youtube.com/watch?v=IsE6g05ioVI)

## Install

```
import Pkg
Pkg.add(url = "https://github.com/PharmCat/GPUCellListSPH.jl.git")
```

## Using 

```julia
using BenchmarkTools, GPUCellListSPH, CUDA

cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 2)))

system = GPUCellListSPH.GPUCellList(cpupoints, (0.016, 0.016), 0.016)

system.points # points

system.pairs # pairs list

system.grid # cell grid 

sum(system.cellpnum) # total cell number

maximum(system.cellpnum) # maximum particle in cell

count(x-> !iszero(x[1]), system.pairs)  == system.pairsn


GPUCellListSPH.update!(system)

GPUCellListSPH.partialupdate!(system)

count(x-> !iszero(x[1]), system.pairs) == system.pairsn
```

## Benchmark

```julia
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 116 samples with 1 evaluation.
 Range (min … max):  41.310 ms … 47.129 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     43.195 ms              ┊ GC (median):    0.00%        
 Time  (mean ± σ):   43.373 ms ±  1.175 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

        ▁  ▃  ▁ █▁▁ ▁▁▁▃     ▁  ▃        ▃
  ▄▄▇▄▇▁█▄▁█▇▄█▁███▆████▆▇▆▆▆█▇▆█▄▄▄▆▇▁▄▇█▇▄▄▄▁▆▁▁▁▄▁▁▁▁▁▄▁▁▄ ▄
  41.3 ms         Histogram: frequency by time        46.6 ms <

 Memory estimate: 38.81 KiB, allocs estimate: 683.
```

```julia
@benchmark GPUCellListSPH.partialupdate!($system)

BenchmarkTools.Trial: 118 samples with 1 evaluation.
 Range (min … max):  42.290 ms …  43.137 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     42.673 ms               ┊ GC (median):    0.00%        
 Time  (mean ± σ):   42.672 ms ± 167.919 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                     ▁  ▃  ▄ ▁▆ ▁▁█  ▆▁     ▃
  ▆▆▁▁▁▄▆▁▁▆▇▁▆▁▇▁▆▇▇█▆▄█▁▇█▆██▇███▆▆██▁▆▄▆▄█▆▁▁▄▄▄▇▆▆▁▁▁▁▁▁▁▄ ▄
  42.3 ms         Histogram: frequency by time         43.1 ms <

 Memory estimate: 30.70 KiB, allocs estimate: 509.
```

## Simple SPH demo

```julia
using GPUCellListSPH
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
ρ₀  = 1000.0                 
m₀  = ρ₀ * dx * dx
α   = 0.01                  # Artificial viscosity constant
g   = 9.81                  # gravity
c₀  = sqrt(g * 2) * 20      # Speed of sound
γ   = 7                     # Gamma costant, used in the pressure equation of state
Δt  = dt  = 1e-5
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


# batch - number of iteration until check time and vtp
# timeframe - simulation time
# writetime - write vtp file each interval
# path - path to vtp files
# pvc - make paraview collection
# set DPC settings (not used by default)
sphprob.dpc_l₀   = 0.005
sphprob.dpc_λ    = 0.005
sphprob.dpc_pmax = 36000
# set surface tension constant 0 by default
sphprob.s        = 0.05
# set kinematic fluid viscosity constant 0 by default
sphprob.𝜈        = 0.2
# XSPH correction constant 0 by default
xsph_𝜀           = 0.5

timesolve!(sphprob; batch = 10, timeframe = 1.0, writetime = 0.02, path = "D:/vtk/", pvc = true)

# timestepping adjust dt
# time lims for dt
# now Δt adjust often buggy
#timesolve!(sphprob; batch = 10, timeframe = 10.0, writetime = 0.02, vtkpath = "D:/vtk/", pvc = true, timestepping = true, timelims = (-Inf, +Inf)) 
```

## Acknowledgment

 * [AhmedSalih3d](https://github.com/AhmedSalih3d)