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


    dx  = 0.02
    h   = 1.2 * sqrt(2) * dx
    H   = 2h
    h⁻¹ = 1/h
    H⁻¹ = 1/H
    dist = H
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


    # (1, 4867, 0.06) # r 

    h     = H * 0.5
    h⁻¹   = 1 / h

    d  =  sqrt((cpupoints[1][1] -  cpupoints[4867][1])^2 + (cpupoints[1][2] -  cpupoints[4867][2])^2)
    αD  = (7/(4 * π * H^2)) # <<
    SPHExample.Wᵢⱼ(αD, d/2H) 


    αD  = (7/( π * H^2 * 2))
    SPHExample.Wᵢⱼ(αD, d/2H)


    sphn = sphkernel.norm * (1/2H)^sphkernel.dim

    val = 𝒲(sphkernel, d/2H, 1/2H)
    
    
    
    αD  = (7/( π * h^2 * 16))

    αD  = 7/π * h⁻¹^2
    SPHExample.Wᵢⱼ(1/H^2, d/H) * 7/π
    
    val = 𝒲(sphkernel, d/2H, 1/2H)

    sphn = sphkernel.norm * h⁻¹^sphkernel.dim
    t1 = 1 - u
    t4 = t1 * t1 * t1 * t1
    (t4 * (1 + 4u)) * sphn



    #CUDA.@device_code_typed GPUCellListSPH.∑ⱼWᵢⱼ!(sumW, cellcounter, pairs, sphkernel, h⁻¹)

    @benchmark  GPUCellListSPH.∑W_2d!($copy(sumW), $cellcounter, $pairs, $sphkernel, $h⁻¹)

    @benchmark GPUCellListSPH.∑∇W_2d!($copy(sum∇W), $∇Wₙ, $cellcounter, $pairs, $points, $sphkernel, $h⁻¹) 


    dx  = 0.02
    H   = 1.2 * sqrt(2) * dx
    h   = H/2
    h⁻¹ = 1/h
    dist = 2H
    system = GPUCellListSPH.GPUCellList(cpupoints, (H, H), H)
    GPUCellListSPH.update!(system)

    @benchmark GPUCellListSPH.update!($system)
    @benchmark GPUCellListSPH.partialupdate!($system)

    #=
    using SPHKernels
    sphk     = WendlandC6(Float64, 3)
    r     = 0.5
    h     = 1.0
    h_inv = 1.0 / h
    u     = r * h_inv
    val = 𝒲(sphk, u, h_inv)
    d𝒲(sphk, u, h_inv)

    SPHKernels.∇𝒲
    =#

    using SPHKernels, StaticArrays
    sphk     = WendlandC2(Float64, 2)
    r     = 0.5
    h     = 1.0
    h_inv = 1.0 / h
    u     = r * h_inv
    val = 𝒲(sphk, u, h_inv)

    Δx    = SVector((0.1,0.1,0.1))
    ∇𝒲(sphk, r, h⁻¹, Δx)


    val = 𝒲(sphk, u, h_inv)

    d𝒲(sphk, u, h_inv)

    ∇𝒲(sphk, r, h⁻¹, Δx)

function sdf(list, pairs)
    inds = Int[]
    for i = 1:length(list)
        el1 = findfirst(x-> x[1] == list[i][1] && x[2] == list[i][2], pairs)
        el2 = findfirst(x-> x[2] == list[i][1] && x[1] == list[i][2], pairs)
        if isnothing(el1) && isnothing(el2) push!(inds, i) end
    end
    inds
end
sdfinds = sdf(list, Array(pairs))


function btpn(cpupoints, dist)
    n = 0
    for i = 1:length(cpupoints)-1
        for j = i+1:length(cpupoints)
            if sqrt((cpupoints[i][1] -  cpupoints[j][1])^2 + (cpupoints[i][2] -  cpupoints[j][2])^2) < dist 
                n += 1 
            end
        end
    end
    n
end
btpn(cpupoints, dist)




    d  = 0.2
    H  = 0.3
    αD  = (7/(4 * π * H^2)) 
    SPHExample.Wᵢⱼ(αD, d/2H) 


    using SPHKernels
    sphkernel    = WendlandC2(Float64, 2)
    𝒲(sphkernel, d/2H, 1/2H)


    using SPHKernels
    function Wᵢⱼ(αD, q)
        return αD * (1 - q) ^ 4 * (1 + 4q)
    end
    d    = 0.2
    H    = 0.3
    αD  = (7/(4 * π * H^2)) 
    w1 =  Wᵢⱼ(αD, d/2H) 
    sphkernel    = WendlandC2(Float64, 2)
    w2 =   𝒲(sphkernel, d/2H, 1/2H)
    w1 ≈ w2

