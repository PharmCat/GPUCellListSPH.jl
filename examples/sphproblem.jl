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
h   = 1.2 * sqrt(2) * dx    # Smoothinl length
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

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float32, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)


sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, ptype, ρ₀, m₀, Δt, α, c₀, γ, δᵩ, CFL; s = 0.0)

stepsolve!(sphprob, 1)

stepsolve!(sphprob, 5000)


get_points(sphprob)

get_velocity(sphprob)

get_density(sphprob)

get_acceleration(sphprob)


@benchmark stepsolve!($sphprob, 100)


#@benchmark stepsolve!($sphprob, 1; simwl = GPUCellListSPH.Effective())