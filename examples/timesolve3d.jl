using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/3D_DamBreak_Fluid_3LAYERS.csv")
boundary_csv = joinpath(path, "../test/input/3D_DamBreak_Boundary_3LAYERS.csv")
DF_FLUD       = CSV.File(fluid_csv) |> DataFrame
DF_FLUD.ptype = fill(Int32(2), size(DF_FLUD, 1))
DF_BOUND      = CSV.File(boundary_csv) |> DataFrame
DF_BOUND.ptype = fill(Int32(1), size(DF_BOUND, 1))
DF_POINTS = append!(DF_FLUD, DF_BOUND)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2", "Points:1"]])...)
cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2", "Points:1"]]))...)
dx  = 0.0085
h   = sqrt(3dx^2)
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = 1.1H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx * dx
α   = 0.01
g   = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (dist, dist, dist)
sphkernel    = WendlandC2(Float32, 3)

system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float32, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)


sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, ptype, ρ₀, m₀, Δt, α, c₀, γ, δᵩ, CFL; s = 0.0)


# batch - number of iteration until check time and vtp
# timeframe - simulation time
# vtkwritetime - write vtp file each intervalgr()
# vtkpath - path to vtp files
# pcx - make paraview collection
sphprob.dpc_l₀   = 0.0005
sphprob.dpc_λ    = 0.05
sphprob.dpc_pmax = 36000
sphprob.s        = 0.0
sphprob.𝜈        = 0.0
sphprob.xsph_𝜀   = 0.0
sphprob.bound_D  = 0.35
sphprob.bound_l  = 1.6dx
plotsettings = Dict(:leg => false, :zlims => (0.3, 0.5))

timesolve!(sphprob; batch = 16, timeframe = 4.0, writetime = 0.001, path = "D:/vtk/", pvc = true)