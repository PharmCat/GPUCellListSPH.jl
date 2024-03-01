using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")
DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = Tuple.(eachrow(DF_POINTS[!, ["Points:0", "Points:2"]]))

dx  = 0.02
h   = 1.2 * sqrt(2) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = 1.1H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx
α   = 0.01
g   = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (dist, dist)
sphkernel    = WendlandC2(Float64, 2)

system  = GPUCellList(cpupoints, cellsize, dist)
N       = length(cpupoints)
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)
v       = CUDA.fill((0.0, 0.0), length(cpupoints))

sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, v, ptype, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)

stepsolve!(sphprob, 1)

stepsolve!(sphprob, 1000)


get_points(sphprob)

get_velocity(sphprob)

get_density(sphprob)

get_acceleration(sphprob)


@benchmark stepsolve!($sphprob, 100)

#=
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  2.791 s …   2.806 s  ┊ GC (min … max): 0.71% … 0.00%
 Time  (median):     2.799 s              ┊ GC (median):    0.35%
 Time  (mean ± σ):   2.799 s ± 10.694 ms  ┊ GC (mean ± σ):  0.35% ± 0.50%

  █                                                       █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.79 s         Histogram: frequency by time        2.81 s <

 Memory estimate: 76.60 MiB, allocs estimate: 1484501.
=#

#@benchmark stepsolve!($sphprob, 1; simwl = GPUCellListSPH.Effective())