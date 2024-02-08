using BenchmarkTools, GPUCellListSPH, CUDA

cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 2)))

system = GPUCellListSPH.GPUCellList(cpupoints, (0.016, 0.016), 0.016)

system.points # points

system.pairs # pairs list

system.grid # cell grid 

sum(system.cellpnum) # total cell number

maximum(system.cellpnum) # maximum particle in cell

count(x-> !isnan(x[3]), system.pairs)  == system.pairsn


GPUCellListSPH.update!(system)

GPUCellListSPH.partialupdate!(system)

count(x-> !isnan(x[3]), system.pairs) == system.pairsn

@benchmark GPUCellListSPH.update!($system)


@benchmark GPUCellListSPH.partialupdate!($system)

using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "./input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "./input/BoundaryPoints_Dp0.02.csv")


cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv)

    ρ   = cu(Array([DF_FLUID.Rhop;DF_BOUND.Rhop]))
    ml  = cu([ ones(size(DF_FLUID,1)) ; zeros(size(DF_BOUND,1))])
    gf = cu([-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))])
    v   = CUDA.fill((0.0, 0.0), length(cpupoints))
    a   = CUDA.zeros(Float64, length(cpupoints))

    dx  = 0.02
    h   = 1.2 * sqrt(2) * dx
    H   = 2h
    h⁻¹ = 1/h
    H⁻¹ = 1/H
    dist = H
    ρ₀  = 1000
    m₀  = ρ₀ * dx * dx #mᵢ  = mⱼ = m₀
    α   = 0.01
    g   = 9.81
    c₀  = sqrt(g * 2) * 20
    γ   = 7
    dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.2

    cellsize = (H, H)
    gpupoints = cu(cpupoints)
    N      = length(cpupoints)
    pcell = CUDA.fill((Int32(0), Int32(0)), N)
    pvec  = CUDA.zeros(Int32, N)
    cs1 = cellsize[1]
    cs2 = cellsize[2]
    MIN1   = minimum(x->x[1], gpupoints) 
    MIN1   = MIN1 - abs((MIN1 + sqrt(eps())) * sqrt(eps()))
    MAX1   = maximum(x->x[1], gpupoints) 
    MIN2   = minimum(x->x[2], gpupoints) 
    MIN2   = MIN2 - abs((MIN1 + sqrt(eps())) * sqrt(eps()))
    MAX2   = maximum(x->x[2], gpupoints)
    range1 = MAX1 - MIN1
    range2 = MAX2 - MIN2
    CELL1  = ceil(Int, range1/cs1)
    CELL2  = ceil(Int, range1/cs2)

    cellpnum     = CUDA.zeros(Int32, CELL1, CELL2)
    

    GPUCellListSPH.cellmap_2d!(pcell, gpupoints, (cs1, cs2), (MIN1, MIN2))

    GPUCellListSPH.cellpnum_2d!(cellpnum, gpupoints,  (cs1, cs2), (MIN1, MIN2))

    mppcell  = maxpoint = maximum(cellpnum)
    mpair    = GPUCellListSPH.мaxpairs_2d(cellpnum)

    celllist     = CUDA.zeros(Int32, CELL1, CELL2, mppcell)

    fill!(cellpnum, zero(Int32))
    GPUCellListSPH.fillcells_naive_2d!(celllist, cellpnum, pcell)
    
    cnt          = CUDA.zeros(Int, 1)
    pairs        = CUDA.fill((zero(Int32), zero(Int32), NaN), mpair)

    GPUCellListSPH.neib_internal_2d!(pairs, cnt, cellpnum, gpupoints, celllist, dist)

    GPUCellListSPH.neib_external_2d!(pairs, cnt, cellpnum, gpupoints, celllist,  (-1, 1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cnt, cellpnum, gpupoints, celllist,   (0, 1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cnt, cellpnum, gpupoints, celllist,   (1, 1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cnt, cellpnum, gpupoints, celllist,   (1, 0), dist)
#####################################################################
system = GPUCellListSPH.GPUCellList(cpupoints, cellsize, H)
@benchmark GPUCellListSPH.update!($system)
#####################################################################
    sphkernel    = WendlandC2(Float64, 2)

    sumW = CUDA.zeros(Float64, N)

#== ==#

    GPUCellListSPH.∑W_2d!(sumW, cellcounter, pairs, sphkernel, H⁻¹)

    sum∇W = CUDA.zeros(Float64, N, 2)
    ∇Wₙ    =  CUDA.fill((zero(Float64), zero(Float64)), mpair, CELL1, CELL2)
    c∇Wₙ    =  CUDA.fill((zero(Float64), zero(Float64)), mpair, CELL1, CELL2)

    GPUCellListSPH.∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, gpupoints, sphkernel, H⁻¹)
    
    ∇Wₙ2    =  CUDA.fill((NaN, NaN), length(pairs))
    ∇Wₙ3    =  CUDA.fill((NaN, NaN), length(pairs))
    GPUCellListSPH.∑∇W_l_2d!(sum∇W, ∇Wₙ2, cellcounter, pairs, gpupoints, sphkernel, H⁻¹)
    GPUCellListSPH.∑∇W_l2_2d!(sum∇W, c∇Wₙ, cellcounter, pairs, gpupoints, sphkernel, H⁻¹)

    @benchmark GPUCellListSPH.∑∇W_2d!($copy(sum∇W), $∇Wₙ, $cellcounter, $pairs, $gpupoints, $sphkernel, $H⁻¹)
    @benchmark GPUCellListSPH.∑∇W_l_2d!($copy(sum∇W), $∇Wₙ2, $cellcounter, $pairs, $gpupoints, $sphkernel, $H⁻¹)
    @benchmark GPUCellListSPH.∑∇W_l2_2d!($copy(sum∇W), $∇Wₙ, $cellcounter, $pairs, $gpupoints, $sphkernel, $H⁻¹)

    ∑∂ρ∂t = CUDA.zeros(Float64, N)

    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml) 

    ∑∂Π∂t = CUDA.zeros(Float64, N, 2)
    

    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, gpupoints, h, ρ, α, v, c₀, m₀)
    
    ∑∂v∂t = CUDA.zeros(Float64, N, 2)

    GPUCellListSPH.∂v∂t!(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, m₀, ρ, c₀, γ, ρ₀) 



    #CUDA.registers(@cuda GPUCellListSPH.kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml))