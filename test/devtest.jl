using CUDA, BenchmarkTools

N      = 200000
CELL   = 60
MPCELL = 100 
MPPAIR = 9000

points       = cu(map(x->tuple(x...), eachrow(rand(Float64, N, 3))))
pcell        = CUDA.fill((Int32(0), Int32(0)), N)
pvec         = CUDA.zeros(Int32, N)
cellpnum     = CUDA.zeros(Int32, CELL, CELL)
cellcounter  = CUDA.zeros(Int32, CELL, CELL)

celllist     = CUDA.zeros(Int32, CELL, CELL, MPCELL);

pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), MPPAIR, CELL, CELL)

sqeps  = sqrt(eps())
MIN1   = minimum(x->x[1], points) 
MAX1   = maximum(x->x[1], points) 
MIN2   = minimum(x->x[2], points) 
MAX2   = maximum(x->x[2], points) 
range1 = MAX1 - MIN1
range2 = MAX2 - MIN2
MIN1 -= sqeps
MIN2 -= sqeps
h1 = range1/CELL + sqeps
h2 = range2/CELL + sqeps
h = (h1, h2)
offset = (MIN1, MIN2)

dist = h1


cellmap_2d!(pcell, points, h, offset)
minimum(x->x[1], pcell) == 1
minimum(x->x[2], pcell) == 1
maximum(x->x[1], pcell) == CELL
maximum(x->x[2], pcell) == CELL

@benchmark cellmap_2d!(pcell, points, h, offset)

cellpnum = CUDA.zeros(Int32, CELL, CELL)
cellpnum_2d!(cellpnum, points,  h, offset)
sum(cellpnum) == N 

@benchmark cellpnum_2d!($copy(cellpnum), points,  h, offset)


cellpnum = CUDA.zeros(Int32, CELL, CELL)
celllist = CUDA.zeros(Int32, CELL, CELL, MPCELL);
fill!(cellcounter, 0)
cellmap_2d!(pcell, points, h, offset)
fillcells_cspwn_2d!(celllist, cellcounter,  pcell) 
sum(cellpnum) == N
count(x-> x > 0, celllist) == N
maximum(cellpnum) <= MPCELL

cellpnum = CUDA.zeros(Int32, CELL, CELL)
celllist = CUDA.zeros(Int32, CELL, CELL, MPCELL);
@benchmark fillcells_cspwn_2d!($copy(celllist), $copy(cellpnum),  $pcell)


fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))

cellmap_2d!(pcell, points, h, offset)
fillcells_cspwn_2d!(celllist, cellpnum,  pcell)
neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, dist)

maximum(cellcounter) <= MPPAIR
count(x-> !isnan(x[3]), pairs) == sum(cellcounter)


fill!(cellcounter, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
@benchmark neib_internal_2d!($copy(pairs), $copy(cellcounter), cellpnum, celllist, points, dist)


fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))

cellmap_2d!(pcell, points, h, offset)
fillcells_cspwn_2d!(celllist, cellpnum,  pcell)
neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, dist)

neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (0, 1), dist)
maximum(cellcounter) <= MPPAIR
count(x-> !isnan(x[3]), pairs) == sum(cellcounter)

neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1, 0), dist)
maximum(cellcounter) <= MPPAIR
count(x-> !isnan(x[3]), pairs) == sum(cellcounter)

neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1, 1), dist)
maximum(cellcounter) <= MPPAIR
count(x-> !isnan(x[3]), pairs) == sum(cellcounter)

fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)
fillcells_cspwn_2d!(celllist, cellpnum,  pcell)
neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, dist)

@benchmark neib_external_2d!($copy(pairs), $copy(cellcounter), cellpnum, points, celllist,  (0, 1), dist)

fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)

sortperm!(pvec, pcell; by=first)
cellthreadmap_2d!(celllist, cellpnum, pvec, pcell) 

sum(cellpnum) == N
count(x-> x > 0, celllist) == N
maximum(cellpnum) <= MPCELL

fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)
fillcells_psort_2d!(celllist, cellpnum, pvec, pcell)

fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)
@btime fillcells_psort_2d!($copy(celllist), $copy(cellpnum), pvec, pcell)


fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)
fillcells_naive_2d!(celllist, cellcounter,  pcell) 



using BenchmarkTools

cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 3)))


system = GPUCellListSPH.GPUCellList(cpupoints, (0.016, 0.016), 0.016)

system.points # points

system.pairs # pairs for each cell

system.grid # cell grid 

sum(system.cellpnum) # total cell number

maximum(system.cellpnum) # maximum particle in cell

maximum(system.cellcounter) # maximum pairs in cell

count(x-> !isnan(x[3]), system.pairs) == sum(system.cellcounter)

GPUCellListSPH.update!(system)

count(x-> !isnan(x[3]), system.pairs) == sum(system.cellcounter)

@benchmark GPUCellListSPH.update!($system)


@benchmark GPUCellListSPH.partialupdate!($system)

using GPUCellListSPH
using CSV, DataFrames, CUDA
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
    cellcounter  = CUDA.zeros(Int32, CELL1, CELL2)

    GPUCellListSPH.cellmap_2d!(pcell, gpupoints, (cs1, cs2), (MIN1, MIN2))

    GPUCellListSPH.cellpnum_2d!(cellpnum, gpupoints,  (cs1, cs2), (MIN1, MIN2))

    mppcell  = maxpoint = maximum(cellpnum)
    mpair    = maxpoint^2*3
    mpair    = 300

    celllist     = CUDA.zeros(Int32, CELL1, CELL2, mppcell)

    #fillcells_cspwn_2d!(celllist, cellcounter,  pcell)
    GPUCellListSPH.fillcells_naive_2d!(celllist, cellcounter,  pcell)

    pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), mpair, CELL1, CELL2)
    fill!(cellcounter, zero(Int32))
    GPUCellListSPH.neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, gpupoints, dist)
    GPUCellListSPH.neib_external_2d!(pairs, cellcounter, cellpnum, gpupoints, celllist,  (1, -1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cellcounter, cellpnum, gpupoints, celllist,  (0, 1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cellcounter, cellpnum, gpupoints, celllist,  (1, 1), dist)
    GPUCellListSPH.neib_external_2d!(pairs, cellcounter, cellpnum, gpupoints, celllist,  (1, 0), dist)

    sphkernel    = WendlandC2(Float64, 2)

    sumW = CUDA.zeros(Float64, N)

    GPUCellListSPH.∑W_2d!(sumW, cellcounter, pairs, sphkernel, H⁻¹)

    sum∇W = CUDA.zeros(Float64, N, 2)
    ∇Wₙ    =  CUDA.fill((zero(Float64), zero(Float64)), mpair, CELL1, CELL2)

    GPUCellListSPH.∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, gpupoints, sphkernel, H⁻¹) 

    ∑∂ρ∂t = CUDA.zeros(Float64, N)

    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml) 

    ∑∂Π∂t = CUDA.zeros(Float64, N, 2)
    

    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, gpupoints, h, ρ, α, v, c₀, m₀)
    
    ∑∂v∂t = CUDA.zeros(Float64, N, 2)

    GPUCellListSPH.∂v∂t!(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, m₀, ρ, c₀, γ, ρ₀) 



    #CUDA.registers(@cuda GPUCellListSPH.kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml))


    # (1, 4867, 0.06) # r 
    list[1] # 1 4707
    ind1 = findfirst(x-> (x[1] == 1 && x[2] == 4707) || (x[2] == 1 && x[1] == 4707), pairs) #
    pair  = pairs[34, 1, 1] # 4707 1
    pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
    xᵢ    = cpupoints[pᵢ]
    xⱼ    = cpupoints[pⱼ]
    u     = d * H⁻¹
    dwk_r = d𝒲(sphkernel, u, H⁻¹) / d
    ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
    ∇Wₙ[34, 1, 1]
    WgL[1]

    list[2] # 1 4709
    ind1 = findfirst(x-> (x[1] == 1 && x[2] == 4709) || (x[2] == 1 && x[1] == 4709), pairs) #
    pair  = pairs[87, 1, 1] # 4709 1
    pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
    xᵢ    = cpupoints[pᵢ]
    xⱼ    = cpupoints[pⱼ]
    u     = d * H⁻¹
    dwk_r = d𝒲(sphkernel, u, H⁻¹) / d
    ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
    ∇Wₙ[34, 1, 1]
    WgL[1]


    list[20] # 4705 4713
    ind1 = findfirst(x-> (x[1] == 4705 && x[2] == 4713) || (x[2] == 4713 && x[1] == 4705), pairs) #
    pair  = pairs[4, 1, 1] # 4705 4713
    pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
    xᵢ    = cpupoints[pᵢ]
    xⱼ    = cpupoints[pⱼ]
    u     = d * H⁻¹
    dwk_r = d𝒲(sphkernel, u, H⁻¹) / d
    ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
    ∇Wₙ[4, 1, 1]
    WgL[20]


    function collctgrad(sum∇W, ∇Wₙ, pairs)
        pairs = collect(pairs)
        ∇Wₙ = collect(∇Wₙ)
        sum∇W = zeros(Float64, N, 2)
        for (k, v) in enumerate(∇Wₙ)
            p1, p2, d = pairs[k]
            if p1 > 0 && p2 > 0
                gr        = ∇Wₙ[k]
                sum∇W[p1, 1] += gr[1]
                sum∇W[p1, 2] += gr[2]
                sum∇W[p2, 1] -= gr[1]
                sum∇W[p2, 2] -= gr[2]
            end
        end
        sum∇W
    end
    res = collctgrad(sum∇W, ∇Wₙ, pairs)
