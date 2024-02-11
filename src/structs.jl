mutable struct GPUCellList
    n::Int
    dist::Float64
    cs
    offset
    grid
    points::CuArray
    pcell::CuArray
    pvec::CuArray
    cellpnum::CuArray
    cnt::CuArray
    celllist::CuArray
    pairs::CuArray
    pairsn::Int
end

"""
    GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)

Make cell list structure.
"""
function GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)
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
    CELL2  = ceil(Int, range2/cs2)                              # number of cells 2-dim

    cellpnum     = CUDA.zeros(Int32, CELL1, CELL2)              # 2-dim array for number of particles in each cell 
    cnt          = CUDA.zeros(Int, 1)                           # temp array for particles counter (need to count place for each pair in pair list)
    points       = cu(points)                                   # array with particles / points

    #cellmap_2d!(pcell, points, (cs1, cs2), (MIN1, MIN2))        # modify pcell < assign cell to each particle

    # if particle out of range it placed it first or in last cell
    # other side in some circumstances better not to include "outranged" particles in list
    cellmap_2d!(pcell, cellpnum, points,  (cs1, cs2), (MIN1, MIN2))                 # modify pcell, cellpnum < pcell - map each point to cell, cellpnum - number of particle in each cell

    maxpoint = Int(ceil(maximum(cellpnum)*1.05 + 1))                                # mppcell - maximum particle in cell for cell list (with reserve ~ 5%)
    if mppcell < maxpoint mppcell = maxpoint end
    
    
    celllist     = CUDA.zeros(Int32, mppcell, CELL1, CELL2)                         # cell list - 3-dim array, 1-dim X 2-dim - cell grid, 3-dim - particle list

    # fillcells_cspwn_2d! or fillcells_psort_2d! function can be used
    # but seems fillcells_naive_2d! is faster
    fill!(cellpnum, Int32(0))                                                          # set cell counter to zero 
    fillcells_naive_2d!(celllist, cellpnum,  pcell)                                    # modify cellcounter, cellpnum < fill cell list by particles; cellpnum should be all zeros!


    if mpairs == 0
        mpairs = мaxpairs_2d(cellpnum)                                                 # mpairs - maximum pairs in pair list (all combination inside cell and neighboring cells (4))
    end
    
    pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), mpairs)                      # pair list
    fill!(cnt, zero(Int32))                                                            # fill cell pairs counter before neib calk
    neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)                    # modify cnt, pairs < add pairs inside cell
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  (1, -1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (-1, 1) in grid
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  (0,  1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (0, 1) in grid
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  (1,  1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (1, 1) in grid
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  (1,  0), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (0, 1) in grid
                                                                                       # now cnt[1] - total number of pairs 
    total_pairs = CUDA.@allowscalar cnt[1]                                                                                 
    if total_pairs > mpairs                                                                    # if calculated pairs num < list length - make a warn
        @warn "List of pairs not full, call `update!(c::GPUCellList)` to extend pair list!"
    end
    if total_pairs < mpairs * 0.8                                                              # if calculated pairs num < 0.8 list length - make  new list with 20% additional pairs
        new_pairn  = Int(ceil(total_pairs / 0.8))
        new_pairs  = CUDA.fill((zero(Int32), zero(Int32), NaN), new_pairn)
        copyto!(new_pairs, view(pairs, 1:new_pairn))
        CUDA.unsafe_free!(pairs)
        pairs      = new_pairs
    end
    GPUCellList(N, dist, cellsize, (MIN1, MIN2), (CELL1, CELL2), points, pcell, pvec, cellpnum, cnt, celllist, pairs, total_pairs)
end

"""
    update!(c::GPUCellList)

Full update cell grid.
"""
function update!(c::GPUCellList)
    #cellmap_2d!(c.pcell, c.points, (c.cs[2], c.cs[2]), c.offset)

    fill!(c.cellpnum, zero(Int32))
    cellmap_2d!(c.pcell, c.cellpnum, c.points, (c.cs[2], c.cs[2]), c.offset)     # modify pcell, cellpnum < pcell - map each point to cell, cellpnum - number of particle in each cell
    maxpoint = maximum(c.cellpnum)

    mppcell, CELLX, CELLY = size(c.celllist)

    if maxpoint > mppcell                                                        # extend cell list if not enough space
        mppcell = Int(ceil(maxpoint*1.05 + 1)) 
        #CUDA.unsafe_free!(c.celllist)
        c.celllist = CUDA.zeros(Int32, mppcell, CELLX, CELLY)
    else
        fill!(c.celllist, zero(Int32))                                           # or fill zeros
    end

    fill!(c.cellpnum, zero(Int32))                                               # or fill zeros before each  "fillcells"

    fillcells_naive_2d!(c.celllist, c.cellpnum,  c.pcell)

    fill!(c.cnt, zero(Int32))

    if c.pairsn > length(c.pairs) || c.pairsn < length(c.pairs) * 0.6                       # if current number of pairs more than pair list or too small - then resize
        CUDA.unsafe_free!(c.pairs)
        c.pairs    = CUDA.fill((zero(Int32), zero(Int32), NaN), Int(ceil(c.pairsn/0.8))) 
    else
        fill!(c.pairs, (zero(Int32), zero(Int32), NaN))
    end

    neib_internal_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist, c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1, -1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (0,  1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1,  1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1,  0), c.dist)

    c.pairsn = CUDA.@allowscalar c.cnt[1]  
end

"""
    partialupdate!(c::GPUCellList)

Update only distance 
"""
function partialupdate!(c::GPUCellList)
    fill!(c.cnt, zero(Int32))
    fill!(c.pairs, (zero(Int32), zero(Int32), NaN))
    neib_internal_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist, c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1, -1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (0,  1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1,  1), c.dist)
    neib_external_2d!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist,  (1,  0), c.dist)
    c.pairsn = CUDA.@allowscalar c.cnt[1]  
end

"""
    neighborlist(c::GPUCellList)

List of pairs with distance.
"""
function neighborlist(c::GPUCellList)
    c.pairs
end


function Base.show(io::IO, c::GPUCellList)
    println(io, "    Cell List")
    println(io, "  Points number: ", c.n)
    println(io, "  Pairs number: ", c.pairsn)
    println(io, "  Cell grid: ", c.grid[1], "X", c.grid[2])
    println(io, "  Cutoff: ", c.dist)
end

