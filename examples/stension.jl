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
g   = 0.0
s   = 0.03
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (H, H)
sphkernel    = WendlandC2(Float64, 2)
cpupoints    = cpupoints[1:4704]
system  = GPUCellList(cpupoints, cellsize, H)
N       = length(cpupoints)
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_FLUID.Rhop)

ml        = CUDA.ones(Float64, N)

isboundary  = .!Bool.(ml)

gf        = CUDA.zeros(Float64, N)
copyto!(gf,-ones(size(DF_FLUID,1)) )

v           = CUDA.fill((0.0, 0.0), length(cpupoints))

sphprob =  SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL; s = s)

prob = sphprob

# batch - number of iteration until check time and vtp
# timeframe - simulation time
# vtkwritetime - write vtp file each interval
# vtkpath - path to vtp files
# pcx - make paraview collection
timesolve!(sphprob; batch = 10, timeframe = 20.0, vtkwritetime = 0.01, vtkpath = "D:/vtk/", pvc = true, timestepping = false)


