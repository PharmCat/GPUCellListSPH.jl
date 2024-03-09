using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")

DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2"]])...)
#cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2"]]))...)
dx  = 0.02
h   = sqrt(2 * dx^2) 
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = 1.1H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx
α   = 0.01
s   = 0.01
g = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (dist, dist)
sphkernel    = WendlandC2(Float64, 2)

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)

sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, ptype, ρ₀, m₀, Δt, α, c₀, γ, δᵩ, CFL; s = 0.0)


# batch - number of iteration until check time and vtp
# timeframe - simulation time
# vtkwritetime - write vtp file each intervalgr()
# vtkpath - path to vtp files
# pcx - make paraview collection
sphprob.dpc_l₀   = 0.001
sphprob.dpc_λ    = 0.005
sphprob.dpc_pmax = 36000
sphprob.s        = 0.01
sphprob.𝜈        = 0.01
sphprob.xsph_𝜀   = 0.0005
timesolve!(sphprob; batch = 100, timeframe = 1.1, writetime = 0.0025, path = "D:/vtk/", pvc = true, anim = true, 
plotsettings = Dict(:leg => false, :xlims => (0, 4), :ylims => (0, 3.5)))

#makedf(sphprob)