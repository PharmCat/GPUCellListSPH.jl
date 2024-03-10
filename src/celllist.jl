mutable struct GPUCellList{T <: AbstractFloat, D}
    n::Int
    dist::T
    cs::Tuple{Vararg{T}}
    offset::Tuple{Vararg{T}}
    grid::Tuple{Vararg{Int}}
    points::NTuple{D, CuArray{T}} 
    pcell::CuArray{NTuple{D, Int32}}
    #pvec::CuArray{Int32}
    cellpnum::CuArray{Int32}
    cnt::CuArray
    celllist::CuArray{Int32}
    pairs::CuArray{Tuple{Int32, Int32}}
    pairsn::Int
end

"""
    GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)

Make cell list structure.
"""
function GPUCellList(points::NTuple{D, AbstractArray{T}}, cellsize, dist; MIN = nothing, MAX = nothing, mppcell = 0, mpairs = 0) where T <: AbstractFloat where D
    pl = length.(points)
    if !all(x->x==first(pl), pl) error("length of points not equal") end

    if D < 2 || D > 3 error("wrong dimention") end
    cellsize = T.(cellsize)
    dist = T(dist)
    N = length(first(points))                                    # Number of points 
    pcell = CUDA.fill(Tuple(fill(Int32(0), D)), N)                   # list of cellst for each particle
    #pvec  = CUDA.zeros(Int32, N)                                # vector for sorting method fillcells_psort_2d!
    for i in eachindex(cellsize)                                 # cell size by 2-dim 
        if cellsize[i] < dist 
            @warn "Cell size (dim $i) < dist, cell size set equal dist"
            csa      = collect(cellsize)
            csa[i]   = dist 
            cellsize = Tuple(csa)
        end
    end

    if isnothing(MIN) MIN = minimum.(points)  end                # minimal value 
    MIN    = @. MIN - abs((MIN + sqrt(eps())) * sqrt(eps()))     # minimal value  (a lillte bit less for better cell fitting)
    if isnothing(MAX) MAX = maximum.(points) end              # maximum                           
    range  = MAX .- MIN                                          # range

    CELL   = @. ceil(Int, range/cellsize)                        # number of cells 

    cellpnum     = CUDA.zeros(Int32, CELL...)                    # array for number of particles in each cell 
    cnt          = CUDA.zeros(Int32, 1)                          # temp array for particles counter (need to count place for each pair in pair list)
    points       = CuArray{T}.(points)                                   # array with particles / points

    
    # if particle out of range it placed it first or in last cell
    # other side in some circumstances better not to include "outranged" particles in list
    cellmap!(pcell, cellpnum, points, cellsize, MIN)                 # modify pcell, cellpnum < pcell - map each point to cell, cellpnum - number of particle in each cell

    maxpoint = Int(ceil(maximum(cellpnum)*1.05 + 1))                                # mppcell - maximum particle in cell for cell list (with reserve ~ 5%)
    if mppcell < maxpoint mppcell = maxpoint end
    
    
    celllist     = CUDA.zeros(Int32, mppcell, CELL...)                         # cell list - 3-dim array, 1-dim X 2-dim - cell grid, 3-dim - particle list

    # fillcells_cspwn_2d! or fillcells_psort_2d! function can be used
    # but seems fillcells_naive_2d! is faster
    fill!(cellpnum, Int32(0))                                                          # set cell counter to zero 
    fillcells_naive!(celllist, cellpnum,  pcell)                                    # modify cellcounter, cellpnum < fill cell list by particles; cellpnum should be all zeros!


    if mpairs == 0
        mpairs = мaxpairs(cellpnum)                                                 # mpairs - maximum pairs in pair list (all combination inside cell and neighboring cells (4))
    end
    
    pairs    = CUDA.fill((zero(Int32), zero(Int32)), mpairs)                      # pair list
    neib_search!(pairs, cnt, cellpnum, points, celllist, dist)   
                                                                                       # now cnt[1] - total number of pairs 
    total_pairs = CUDA.@allowscalar cnt[1]                                                                                 
    if total_pairs > mpairs                                                                    # if calculated pairs num < list length - make a warn
        @warn "List of pairs not full, call `update!(c::GPUCellList)` to extend pair list!"
    end
    if total_pairs < mpairs * 0.8                                                              # if calculated pairs num < 0.8 list length - make  new list with 20% additional pairs
        new_pairn  = Int(ceil(total_pairs / 0.8))
        new_pairs  = CUDA.fill((zero(Int32), zero(Int32)), new_pairn)
        copyto!(new_pairs, view(pairs, 1:new_pairn))
        CUDA.unsafe_free!(pairs)
        pairs      = new_pairs
    end
    GPUCellList{T, D}(N, dist, cellsize, MIN, CELL, points, pcell, cellpnum, cnt, celllist, pairs, total_pairs)
end

"""
    update!(c::GPUCellList)

Full update cell grid.
"""
@noinline function update!(c::GPUCellList, fillzero::Bool = true)
    #cellmap_2d!(c.pcell, c.points, (c.cs[2], c.cs[2]), c.offset)

    fill!(c.cellpnum, zero(Int32))
    cellmap!(c.pcell, c.cellpnum, c.points, c.cs, c.offset)     # modify pcell, cellpnum < pcell - map each point to cell, cellpnum - number of particle in each cell
    maxpoint = maximum(c.cellpnum)

    mppcell = size(c.celllist, 1)
    CELL    = size(c.celllist)[2:end]

    if maxpoint > mppcell                                                        # extend cell list if not enough space
        mppcell = Int(ceil(maxpoint*1.05 + 1)) 
        CUDA.unsafe_free!(c.celllist)
        c.celllist = CUDA.zeros(Int32, mppcell, CELL...)
    else
        fill!(c.celllist, zero(Int32))                                           # or fill zeros
    end

    fill!(c.cellpnum, zero(Int32))                                               # or fill zeros before each  "fillcells"

    fillcells_naive!(c.celllist, c.cellpnum,  c.pcell)

    if c.pairsn > length(c.pairs) || c.pairsn < length(c.pairs) * 0.6                       # if current number of pairs more than pair list or too small - then resize
        CUDA.unsafe_free!(c.pairs)
        c.pairs    = CUDA.fill((zero(Int32), zero(Int32)), Int(ceil(c.pairsn/0.8))) 
    elseif fillzero
        fill!(c.pairs, (zero(Int32), zero(Int32)))
    end

    neib_search!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist, c.dist) 

    CUDA.@allowscalar c.pairsn = c.cnt[1]  

    if c.pairsn > length(c.pairs)
        CUDA.unsafe_free!(c.pairs)
        c.pairs    = CUDA.fill((zero(Int32), zero(Int32)), Int(ceil(c.pairsn/0.8))) 
        neib_search!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist, c.dist) 
        CUDA.@allowscalar c.pairsn = c.cnt[1]  
    end
end

"""
    partialupdate!(c::GPUCellList, fillzero::Bool = true)

Update only distance 
"""
@noinline function partialupdate!(c::GPUCellList, fillzero::Bool = true)
    if fillzero fill!(c.pairs, (zero(Int32), zero(Int32))) end
    neib_search!(c.pairs, c.cnt, c.cellpnum, c.points, c.celllist, c.dist) 
    c.pairsn = CUDA.@allowscalar c.cnt[1]  
end

"""
    neighborlist(c::GPUCellList)

List of pairs with distance.
"""
function neighborlist(c::GPUCellList)
    c.pairs
end

"""
    neighborlistview(c::GPUCellList)

List of pairs with distance.
"""
function neighborlistview(c::GPUCellList)
    view(c.pairs, 1:c.pairsn)
end



function Base.show(io::IO, c::GPUCellList{T, D}) where T where D
    println(io, "    Cell List (Dim: $D)")
    println(io, "  Points number: ", c.n , " Type: $T")
    println(io, "  Pairs number: ", c.pairsn)
    print(io, "  Cell grid: ", c.grid[1], "X", c.grid[2])
    if length(c.grid) >= 3 println(io, "X", c.grid[3]) else println(io, "") end
    println(io, "  Cutoff: ", c.dist)
end

