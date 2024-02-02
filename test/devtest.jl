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

cellmap2d!(pcell, points, h, offset)
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

cellmap2d!(pcell, points, h, offset)
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
cellmap2d!(pcell, points, h, offset)
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




#=
fill!(cellpnum, 0)
fill!(cellcounter, 0)
fill!(celllist, 0)
fill!(pairs, (zero(Int32), zero(Int32), NaN))
cellmap_2d!(pcell, points, h, offset)
fillcells_naiveunsave_2d!(celllist, cellpnum,  pcell) 
=#


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
