

mutable struct GPUCellList
    n::Int
    dist
    cs
    offset
    grid
    points
    pcell
    pvec
    cellpnum
    cellcounter
    celllist
    pairs
end


function GPUCellList(points, cellsize, dist; mppcell = 0, mpair = 0)
    el = first(points)
    if length(el) < 2 error("wrong dimention") end
    N = length(points)
    pcell = CUDA.fill((Int32(0), Int32(0)), N)
    pvec  = CUDA.zeros(Int32, N)
    cs1 = cellsize[1]
    cs2 = cellsize[2]
    MIN1   = minimum(x->x[1], points) 
    MIN1   = MIN1 - abs(MIN1 * sqrt(eps()))
    MAX1   = maximum(x->x[1], points) 
    MIN2   = minimum(x->x[2], points) 
    MIN2   = MIN2 - abs(MIN1 * sqrt(eps()))
    MAX2   = maximum(x->x[2], points)
    range1 = MAX1 - MIN1
    range2 = MAX2 - MIN2
    CELL1  = ceil(Int, range1/cs1)
    CELL2  = ceil(Int, range1/cs2)

    cellpnum     = CUDA.zeros(Int32, CELL1, CELL2)
    cellcounter  = CUDA.zeros(Int32, CELL1, CELL2)
    points       = cu(points)

    cellmap_2d!(pcell, points, (cs1, cs2), (MIN1, MIN2))

    cellpnum_2d!(cellpnum, points,  (cs1, cs2), (MIN1, MIN2))

    maxpoint = maximum(cellpnum)
    if mppcell < maxpoint mppcell = maxpoint end
    if mpair == 0
        mpair = maxpoint^2*3
    end

    celllist     = CUDA.zeros(Int32, CELL1, CELL2, mppcell)

    fillcells_cspwn_2d!(celllist, cellcounter,  pcell) 

    pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), mpair, CELL1, CELL2)
    fill!(cellcounter, zero(Int32))
    neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, dist)
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (0, 1), dist)
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1, 1), dist)
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1, 0), dist)

    GPUCellList(N, dist, cellsize, (MIN1, MIN2), (CELL1, CELL2), points, pcell, pvec, cellpnum, cellcounter, celllist, pairs)
end

function update!(c::GPUCellList)
    cellmap_2d!(c.pcell, c.points, (c.cs[2], c.cs[2]), c.offset)
    fill!(c.cellpnum, zero(Int32))
    cellpnum_2d!(c.cellpnum, c.points, (c.cs[2], c.cs[2]), c.offset)
    fill!(c.celllist, zero(Int32))
    fill!(c.cellcounter, zero(Int32))
    fillcells_cspwn_2d!(c.celllist, c.cellcounter,  c.pcell) 
    fill!(c.cellcounter, zero(Int32))
    neib_internal_2d!(c.pairs, c.cellcounter, c.cellpnum, c.celllist, c.points, c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (0, 1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1, 1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1, 0), c.dist)
end

