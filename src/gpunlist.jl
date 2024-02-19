mutable struct GPUNeighborCellList
    n::Int
    dist::Float64
    cs
    offset
    grid
    points::CuArray
    pcell::CuArray
    cellpnum::CuArray
    cnt::CuArray
    celllist::CuArray
    nlist::CuArray
    max_pairs::Int
end

"""
    GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)

Make cell list structure.
"""
function GPUNeighborCellList(points, cellsize, dist; mppcell = 0, maxneigh = 0)
    el = first(points)
    if length(el) < 2 error("wrong dimention") end

    N = length(points)                                          # Number of points 
    pcell = CUDA.fill((Int32(0), Int32(0)), N)                  # list of cellst for each particle

    cs1 = cellsize[1]                                           # cell size by 1-dim
    cs2 = cellsize[2]                                           # cell size by 2-dim 
    if cs1 < dist 
        @warn "Cell size 1 < dist, cell size set to dist"
         cs1 = dist 
    end
    if cs2 < dist 
        @warn "Cell size 2 < dist, cell size set to dist"
        cs2 = dist 
    end
    MIN1   = minimum(x->x[1], points)                           # minimal value 
    MIN1   = MIN1 - abs((MIN1 + sqrt(eps())) * sqrt(eps()))     # minimal value 1-dim (a lillte bil less for better cell fitting)
    MAX1   = maximum(x->x[1], points)                           # maximum 1-dim
    MIN2   = minimum(x->x[2], points)                           # minimal value 
    MIN2   = MIN2 - abs((MIN2 + sqrt(eps())) * sqrt(eps()))     # minimal value 2-dim (a lillte bil less for better cell fitting)
    MAX2   = maximum(x->x[2], points)                           # maximum 1-dim
    range1 = MAX1 - MIN1                                        # range 1-dim
    range2 = MAX2 - MIN2                                        # range 2-dim
    CELL1  = ceil(Int, range1/cs1)                              # number of cells 1-dim
    CELL2  = ceil(Int, range2/cs2)                              # number of cells 2-dim

    cellpnum     = CUDA.zeros(Int32, CELL1, CELL2)              # 2-dim array for number of particles in each cell 
    cnt          = CUDA.zeros(Int32, N)                         # temp array for particles counter (need to count place for each pair in pair list)
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


    if maxneigh == 0
        maxneigh = maximum(cellpnum)*9                                                 # 
    end

    nlist = CUDA.zeros(Int32, maxneigh, N) 
    
    
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (0, 1))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (0, 0))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (0,-1))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (1, 1))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (1, 0))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist,  (1,-1))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist, (-1, 1))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist, (-1, 0))
    neiblist_2d!(nlist, cnt, points,  celllist, cellpnum, pcell, dist, (-1,-1))
    
    max_pairs = maximum(cnt)                                                                               
    if max_pairs > maxneigh                                                                    # if calculated pairs num < list length - make a warn
        @warn "List of pairs not full, call `update!(c::GPUNeighborCellList)` to extend pair list!"
    end
    GPUNeighborCellList(N, dist, (cs1, cs2), (MIN1, MIN2), (CELL1, CELL2), points, pcell, cellpnum, cnt, celllist, nlist, max_pairs)
end

"""
    update!(c::GPUNeighborCellList)

Full update cell grid.
"""
@noinline function update!(c::GPUNeighborCellList)
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
    fill!(c.nlist, zero(Int32))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0,-1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1,-1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1,-1))
    c.max_pairs = maximum(c.cnt)   

end

"""
    partialupdate!(c::GPUNeighborCellList)

Update only distance 
"""
@noinline function partialupdate!(c::GPUNeighborCellList)
    fill!(c.cnt, zero(Int32))
    fill!(c.nlist, zero(Int32))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (0,-1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist,  (1,-1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1, 1))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1, 0))
    neiblist_2d!(c.nlist, c.cnt, c.points,  c.celllist, c.cellpnum, c.pcell, c.dist, (-1,-1))
    c.max_pairs = maximum(c.cnt)   
end

"""
    neighborlist(c::GPUNeighborCellList)

List of pairs with distance.
"""
function neighborlist(c::GPUNeighborCellList)
    c.nlist
end


function Base.show(io::IO, c::GPUNeighborCellList)
    println(io, "    Neighbor Cell List")
    println(io, "  Points number: ", c.n)
    println(io, "  Maximum pairs number: ", c.max_pairs)
    println(io, "  Cell grid: ", c.grid[1], "X", c.grid[2])
    println(io, "  Cutoff: ", c.dist)
end