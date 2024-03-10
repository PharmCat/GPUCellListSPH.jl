using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/3D_DamBreak_Fluid.csv")
boundary_csv = joinpath(path, "../test/input/3D_DamBreak_Boundary.csv")
DF_FLUD       = CSV.File(fluid_csv) |> DataFrame
DF_FLUD.ptype = ones(Int32, size(DF_FLUD, 1))
DF_BOUND      = CSV.File(boundary_csv) |> DataFrame
DF_BOUND.ptype = fill(Int32(0), size(DF_BOUND, 1))
DF_POINTS = append!(DF_FLUD, DF_BOUND)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2", "Points:1"]])...)

dx  = 0.0085
h   = sqrt(3dx^2)
H   = 2h
dist = 1.1H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx * dx
α   = 0.01
g   = 9.81
s   = 0.01
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (dist, dist, dist)
sphkernel    = WendlandC2(Float64, 3)

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)


sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ,  ptype, ρ₀, m₀, Δt, α,  c₀, γ, δᵩ, CFL; s = 0.0)

stepsolve!(sphprob, 1)

stepsolve!(sphprob, 10)


get_points(sphprob)

get_velocity(sphprob)

get_density(sphprob)

get_acceleration(sphprob)


@benchmark stepsolve!($sphprob, 10)

#=
BenchmarkTools.Trial: 3 samples with 1 evaluation.
 Range (min … max):  2.061 s …   2.084 s  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.065 s              ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.070 s ± 12.433 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █       █                                               █  
  █▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.06 s         Histogram: frequency by time        2.08 s <

 Memory estimate: 1.21 MiB, allocs estimate: 21784.
=#