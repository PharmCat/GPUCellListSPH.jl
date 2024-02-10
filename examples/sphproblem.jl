using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")

cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv)

dx  = 0.02
h   = 1.2 * sqrt(2) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx
α   = 0.01
g   = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (H, H)
sphkernel    = WendlandC2(Float64, 2)

system  = GPUCellListSPH.GPUCellList(cpupoints, cellsize, H)

ρ           = cu(Array([DF_FLUID.Rhop;DF_BOUND.Rhop]))
ml          = cu(append!(ones(Float64, size(DF_FLUID, 1)), zeros(Float64, size(DF_BOUND, 1))))
isboundary  = .!Bool.(ml)
gf          = cu([-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))])
v           = CUDA.fill((0.0, 0.0), length(cpupoints))


sphprob =  SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)

stepsolve!(sphprob, 1000)


get_points(sphprob)

get_velocity(sphprob)

get_density(sphprob)

get_acceleration(sphprob)


@benchmark stepsolve!($sphprob, 1)

#=
BenchmarkTools.Trial: 946 samples with 1 evaluation.
 Range (min … max):  4.714 ms … 42.996 ms  ┊ GC (min … max): 0.00% … 54.74%
 Time  (median):     5.193 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.284 ms ±  1.250 ms  ┊ GC (mean ± σ):  0.47% ±  1.78%

               ▁▃▄▄█▅▆▄▅▂▃▃▁▁
  ▂▁▁▂▂▂▃▄▄▄▄▇▇███████████████▇▅▅▅▄▄▄▄▃▄▄▃▃▄▄▄▃▃▃▂▃▃▃▂▂▂▃▃▃▂ ▄
  4.71 ms        Histogram: frequency by time        6.04 ms <

 Memory estimate: 100.20 KiB, allocs estimate: 1938.
=#