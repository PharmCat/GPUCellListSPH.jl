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

    N = length(points)                                          # Number of points 
    pcell = CUDA.fill((Int32(0), Int32(0)), N)                  # list of cellst for each particle
    pvec  = CUDA.zeros(Int32, N)                                # vector for sorting method fillcells_psort_2d!
    cs1 = cellsize[1]                                           # cell size by 1-dim
    cs2 = cellsize[2]                                           # cell size by 2-dim 
    MIN1   = minimum(x->x[1], points)                           # minimal value 
    MIN1   = MIN1 - abs((MIN1 + sqrt(eps())) * sqrt(eps()))     # minimal value 1-dim (a lillte bil less for better cell fitting)
    MAX1   = maximum(x->x[1], points)                           # maximum 1-dim
    MIN2   = minimum(x->x[2], points)                           # minimal value 
    MIN2   = MIN2 - abs((MIN1 + sqrt(eps())) * sqrt(eps()))     # minimal value 2-dim (a lillte bil less for better cell fitting)
    MAX2   = maximum(x->x[2], points)                           # maximum 1-dim
    range1 = MAX1 - MIN1                                        # range 1-dim
    range2 = MAX2 - MIN2                                        # range 2-dim
    CELL1  = ceil(Int, range1/cs1)                              # number of cells 1-dim
    CELL2  = ceil(Int, range1/cs2)                              # number of cells 2-dim

    cellpnum     = CUDA.zeros(Int32, CELL1, CELL2)              # 2-dim array for number of particles in each cell 
    cellcounter  = CUDA.zeros(Int32, CELL1, CELL2)              # temp 2-dim array for particles counter 
    points       = cu(points)                                   # array with particles / points

    cellmap_2d!(pcell, points, (cs1, cs2), (MIN1, MIN2))        # modify pcell < assign cell to each particle

    cellpnum_2d!(cellpnum, points,  (cs1, cs2), (MIN1, MIN2))   # modify cellpnum < count particle number for each cell

    maxpoint = maximum(cellpnum)                                # mppcell - maximum particle in cell for cell list
    if mppcell < maxpoint mppcell = maxpoint end
    if mpair == 0
        mpair = maxpoint^2*3                                    # mpair - maximum pairs in pair list for each cell
    end

    celllist     = CUDA.zeros(Int32, CELL1, CELL2, mppcell)     # cell list - 3-dim array, 1-dim X 2-dim - cell grid, 3-dim - particle list

    # fillcells_cspwn_2d! or fillcells_psort_2d! function can be used
    # but seems fillcells_naive_2d! is faster
    fillcells_naive_2d!(celllist, cellcounter,  pcell)                                 # modify cellcounter, celllist < fill cell list by particles; cellcounter should be all zeros!

    pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), mpair, CELL1, CELL2)         # pair list 1-dim - pair, 2-dim X 3-dim - cell grid
    fill!(cellcounter, zero(Int32))                                                    # fill cell counter before neib calk
    neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, dist)            # modify cellcounter, celllist < add pairs inside cell
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1, -1), dist) 
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (0,  1), dist)   # modify cellcounter, celllist < add pairs between cell and neiborhood cell by shift (0, 1) in grid
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1,  1), dist)   # modify cellcounter, celllist < add pairs between cell and neiborhood cell by shift (1, 1) in grid
    neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  (1,  0), dist)   # modify cellcounter, celllist < add pairs between cell and neiborhood cell by shift (0, 1) in grid
                                                                                       # now cellcounter - number of pairs for each particle
    GPUCellList(N, dist, cellsize, (MIN1, MIN2), (CELL1, CELL2), points, pcell, pvec, cellpnum, cellcounter, celllist, pairs)
end

"""
    update!(c::GPUCellList)

Full update cell grid.
"""
function update!(c::GPUCellList)
    cellmap_2d!(c.pcell, c.points, (c.cs[2], c.cs[2]), c.offset)
    fill!(c.cellpnum, zero(Int32))
    cellpnum_2d!(c.cellpnum, c.points, (c.cs[2], c.cs[2]), c.offset)
    fill!(c.celllist, zero(Int32))
    fill!(c.cellcounter, zero(Int32))
    #fillcells_cspwn_2d!(c.celllist, c.cellcounter,  c.pcell)
    fillcells_naive_2d!(c.celllist, c.cellcounter,  c.pcell)
    fill!(c.cellcounter, zero(Int32))
    fill!(c.pairs, (zero(Int32), zero(Int32), NaN))
    neib_internal_2d!(c.pairs, c.cellcounter, c.cellpnum, c.celllist, c.points, c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1, -1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (0,  1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1,  1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1,  0), c.dist)
end

"""
    partialupdate!(c::GPUCellList)

Update only distance 
"""
function partialupdate!(c::GPUCellList)
    fill!(c.cellcounter, zero(Int32))
    fill!(c.pairs, (zero(Int32), zero(Int32), NaN))
    neib_internal_2d!(c.pairs, c.cellcounter, c.cellpnum, c.celllist, c.points, c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1, -1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (0,  1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1,  1), c.dist)
    neib_external_2d!(c.pairs, c.cellcounter, c.cellpnum, c.points, c.celllist,  (1,  0), c.dist)
end

