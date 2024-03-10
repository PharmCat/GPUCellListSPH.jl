using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
using Profile
using PProf
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")
DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2"]])...)
#cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2"]]))...)

dx   = 0.02
h    = 1.2 * sqrt(2) * dx
H    = 2h
dist = H
cellsize = (dist, dist)

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