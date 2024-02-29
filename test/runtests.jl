using GPUCellListSPH
using CSV, DataFrames, CUDA
using Test
path         = dirname(@__FILE__)

@testset "GPUCellListSPH.jl" begin
    # Write your tests here.
    
    fluid_csv    = joinpath(path, "./input/FluidPoints_Dp0.02.csv")
    boundary_csv = joinpath(path, "./input/BoundaryPoints_Dp0.02.csv")

    cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv)
    dx  = 0.02
    h   = 1.2 * sqrt(2) * dx
    system = GPUCellListSPH.GPUCellList(cpupoints, (2h, 2h), 2h)
    GPUCellListSPH.update!(system)

end
