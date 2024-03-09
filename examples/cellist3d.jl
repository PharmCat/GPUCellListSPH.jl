using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
using Profile
using PProf
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/3D_DamBreak_Fluid.csv")
boundary_csv = joinpath(path, "../test/input/3D_DamBreak_Boundary.csv")

DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2", "Points:1"]])...)
cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2", "Points:1"]]))...)

dx  = 0.0085
h   = sqrt(3) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = H
cellsize = (dist, dist, dist)

system  = GPUCellList(cpupoints, cellsize, dist)

update!(system)

@benchmark update!($system)

@profile for i=1:1000 update!(system) end

pprof()

#PProf.kill()
#Profile.clear()

#pairs       = GPUCellListSPH.neighborlistview(system)
#sort!(pairs, by = first)
#ranges = CUDA.fill((zero(Int32), zero(Int32)), system.n)
#GPUCellListSPH.pranges!(ranges, pairs) 
#@benchmark GPUCellListSPH.pranges!($ranges, $pairs)