using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)

path = joinpath(dirname(pathof(GPUCellListSPH)))


fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")

cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv) # Load particles 

dx  = 0.02                  # resolution
h   = 1.2 * sqrt(2) * dx    # smoothinl length
H   = 2h                    # kernel support length
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = 1.2H                    # distance for neighborlist
ρ₀  = 1000.0                 
m₀  = ρ₀ * dx * dx
α   = 0.01                  # Artificial viscosity constant
g   = 9.81                  # gravity
c₀  = sqrt(g * 2) * 20      # Speed of sound
γ   = 7                     # Gamma costant, used in the pressure equation of state
Δt  = dt  = 1e-5
δᵩ  = 0.1                   # Coefficient for density diffusion
CFL = 0.2                   # Courant–Friedrichs–Lewy condition for Δt stepping
cellsize = (dist, dist)           # cell size
sphkernel    = WendlandC2(Float64, 2) # SPH kernel from SPHKernels.jl

system  = GPUCellList(cpupoints, cellsize, dist)
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
timesolve!(sphprob; batch = 10, timeframe = 2.0, writetime = 0.025, path = "D:/vtk/", pvc = true)

# timestepping adjust dt
# time lims for dt
# now Δt adjust often buggy
#timesolve!(sphprob; batch = 50, timeframe = 3.5, writetime = 0.01, path = "D:/vtk/", pvc = true, timestepping = true, timelims = (eps(), 1e-5)) 
