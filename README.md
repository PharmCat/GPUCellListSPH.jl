# GPUCellListSPH.jl

## Description

This is concept-project for particle cell neiborhood list and SPH. Based on [AhmedSalih3d](https://github.com/AhmedSalih3d) repo [SPHExample](https://github.com/AhmedSalih3d/SPHExample).


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

count(x-> !isnan(x[3]), system.pairs)  == system.pairsn


GPUCellListSPH.update!(system)

GPUCellListSPH.partialupdate!(system)

count(x-> !isnan(x[3]), system.pairs) == system.pairsn
```

## Benchmark

```julia
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 117 samples with 1 evaluation.
 Range (min … max):  42.519 ms …  43.666 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     42.998 ms               ┊ GC (median):    0.00%        
 Time  (mean ± σ):   42.987 ms ± 230.596 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

             ▃     ▃▁  ▁ ▁   ▁ █ ▃▁▃  ▁
  ▇▄▁▁▄▁▆▄▄▁▆█▄▁▆▇▇██▄▇█▆█▇▆▁█▇█▇███▇▆█▄▇▁▆▁▆▁▄▁▆▁▆▆▁▇▁▁▄▁▄▁▄▄ ▄
  42.5 ms         Histogram: frequency by time         43.5 ms <

 Memory estimate: 40.72 KiB, allocs estimate: 722.
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
using CSV, DataFrames, CUDA, BenchmarkTools

path = joinpath(dirname(pathof(GPUCellListSPH)))

fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")

cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv) # Load particles 

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

system  = GPUCellList(cpupoints, cellsize, H)
N       = length(cpupoints)
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, Array([DF_FLUID.Rhop;DF_BOUND.Rhop]))

ml        = CUDA.zeros(Float64, N)
copyto!(ml, append!(ones(Float64, size(DF_FLUID, 1)), zeros(Float64, size(DF_BOUND, 1))))

isboundary  = .!Bool.(ml)

gf        = CUDA.zeros(Float64, N)
copyto!(gf,[-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))])

v           = CUDA.fill((0.0, 0.0), length(cpupoints))

sphprob =  SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)

# batch - number of iteration until check time and vtp
# timeframe - simulation time
# writetime - write vtp file each interval
# path - path to vtp files
# pvc - make paraview collection
timesolve!(sphprob; batch = 10, timeframe = 1.0, writetime = 0.02, path = "D:/vtk/", pvc = true)

# timestepping adjust dt
# time lims for dt
# now Δt adjust often buggy
#timesolve!(sphprob; batch = 10, timeframe = 10.0, writetime = 0.02, vtkpath = "D:/vtk/", pvc = true, timestepping = true, timelims = (-Inf, +Inf)) 
```

## Acknowledgment

 * [AhmedSalih3d](https://github.com/AhmedSalih3d)