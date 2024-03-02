using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
using Profile
using PProf
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
dist = 1.1*H
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

sphprob =  SPHProblem(system,  dx, h, H, sphkernel, ρ, v, ptype, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL; s = 0.0)

sphprob.dpc_l₀   = 0.005
sphprob.dpc_λ    = 0.005
sphprob.dpc_pmax = 36000
sphprob.s        = 0.05
sphprob.𝜈        = 0.2
xsph_𝜀           = 0.5
stepsolve!(sphprob, 1)

@benchmark stepsolve!(sphprob, 1000)

@profile  stepsolve!(sphprob, 10000)

@profile  timesolve!(sphprob; batch = 300, timeframe = 0.001)

pprof()

PProf.kill()
Profile.clear()