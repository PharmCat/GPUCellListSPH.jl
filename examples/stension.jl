using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
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

s   = 0.05                  # surface tension constant
sphkernel    = WendlandC2(Float64, 2)

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)

ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)

sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, ptype, ρ₀, m₀, Δt, α, c₀, γ, δᵩ, CFL; s = s)

prob = sphprob

# batch - number of iteration until check time and vtp
# timeframe - simulation time
# vtkwritetime - write vtp file each interval
# vtkpath - path to vtp files
# pcx - make paraview collection
timesolve!(sphprob; batch = 10, timeframe = 20.0, vtkwritetime = 0.01, vtkpath = "D:/vtk/", pvc = true, timestepping = false)


