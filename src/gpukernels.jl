#####################################################################
# CELL LIST
#####################################################################

function kernel_cellmap_2d!(pcell, points,  h⁻¹, offset) 
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) * h⁻¹[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) * h⁻¹[2]
        pᵢ₁ = ceil(Int32, p₁)
        pᵢ₂ = ceil(Int32, p₂)
        @inbounds pcell[i] = (pᵢ₁, pᵢ₂) # what to with points outside cell grid?
    end
    return nothing
end
"""
    cellmap_2d!(pcell, points, dist, offset) 
    
Map each point to cell.
"""
function cellmap_2d!(pcell, points, h, offset) 
    h⁻¹ = (1/h[1], 1/h[2])
    kernel = @cuda launch=false kernel_cellmap_2d!(pcell, points,  h⁻¹, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(size(points, 1), config.threads)
    blocks = cld(size(points, 1), threads)
    CUDA.@sync kernel(pcell, points,  h⁻¹, offset; threads = threads, blocks = blocks)
end

#####################################################################

function kernel_cellpnum_2d!(cellpnum, points,  h⁻¹, offset) 
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    csᵢ = size(cellpnum, 1) 
    csⱼ = size(cellpnum, 2) 
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) * h⁻¹[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) * h⁻¹[2]
        pᵢ₁ = ceil(Int32, p₁) 
        pᵢ₂ = ceil(Int32, p₂)
        if pᵢ₁ <= 0  pᵢ₁  = 1  end
        if pᵢ₁ > csᵢ pᵢ₁ = csᵢ end

        if pᵢ₂ <= 0  pᵢ₂  = 1   end
        if pᵢ₂ > csⱼ pᵢ₂  = csⱼ end

        #if csᵢ >= pᵢ₁ > 0 && csⱼ >= pᵢ₂ > 0
            CUDA.@atomic cellpnum[pᵢ₁, pᵢ₂] += one(Int32) 
        #end
    end
    return nothing
end
"""
    Number of points in each cell.
"""
function cellpnum_2d!(cellpnum, points,  h, offset)  
    h⁻¹ = (1/h[1], 1/h[2])
    kernel = @cuda launch=false kernel_cellpnum_2d!(cellpnum, points,  h⁻¹, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(size(points, 1), config.threads)
    blocks = cld(size(points, 1), threads)
    CUDA.@sync kernel(cellpnum, points,  h⁻¹, offset; threads = threads, blocks = blocks)
end

#####################################################################


function kernel_fillcells_naive_2d!(celllist, cellpnum, pcell) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if indexᵢ <= length(pcell)
        csᵢ = size(cellpnum, 1) 
        csⱼ = size(cellpnum, 2) 
        pᵢ, pⱼ = pcell[indexᵢ]

        if pᵢ <= 0  pᵢ  = 1  end
        if pᵢ > csᵢ pᵢ = csᵢ end

        if pⱼ <= 0  pⱼ  = 1   end
        if pⱼ > csⱼ pⱼ  = csⱼ end

        #if n + 1 > size(celllist, 1) || pᵢ > size(celllist, 2) || pⱼ > size(celllist, 3) @cuprintln( n + 1 , " - ", pᵢ ," - ", pⱼ) end

        n = CUDA.@atomic cellpnum[pᵢ, pⱼ] += 1
        
        celllist[n + 1, pᵢ, pⱼ] = indexᵢ
    end
    return nothing
end
"""
    fillcells_naive_2d!(celllist, cellpnum, pcell) 
    
Fill cell list with cell. Naive approach.
"""
function fillcells_naive_2d!(celllist, cellpnum, pcell)  
    CLn, CLx, CLy = size(celllist)
    if size(cellpnum) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_fillcells_naive_2d!(celllist, cellpnum, pcell) 
    config = launch_configuration(gpukernel.fun)
    threads = min(length(pcell), config.threads)
    blocks = cld(length(pcell), threads)
    CUDA.@sync gpukernel(celllist, cellpnum, pcell; threads = threads, blocks = blocks)
end

#####################################################################

#=
"""
    Fill cell list with cell. Each cell iterate over pcell vector.
"""
function kernel_fillcells_cspwn_2d!(celllist, cellcounter,  pcell) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if indexᵢ <= size(celllist, 1) && indexⱼ <= size(celllist, 2)
        n    = cellcounter[indexᵢ, indexⱼ]
        maxn = size(celllist, 3)
        for k = 1:length(pcell)
            @inbounds kind = pcell[k]
            if  kind == (indexᵢ, indexⱼ) 
                n += 1
                if n <= maxn
                    @inbounds celllist[indexᵢ, indexⱼ, n] = k
                end
            end
        end
        cellcounter[indexᵢ, indexⱼ] = n
    end
    return nothing
end
function fillcells_cspwn_2d!(celllist, cellcounter,  pcell)  
    kernel = @cuda launch=false kernel_fillcells_cspwn_2d!(celllist, cellcounter,  pcell) 
    config = launch_configuration(kernel.fun)
    threads = min(size(celllist, 1), Int(floor(sqrt(config.threads))))
    blocks = cld(size(celllist, 1), threads)
    CUDA.@sync kernel(celllist, cellcounter,  pcell; threads = (threads,threads), blocks = (blocks,blocks))
end
=#
#####################################################################
#=
"""
Fill cell list with cell. Each thread find starting point in sorted array 
and iterate only in region for first dimension of cell grid. 
    
"""
function kernel_cellthreadmap_2d!(celllist, cellpnum, pvec, pcell) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if indexᵢ <= size(celllist, 1)
        start = findfirst(x-> pcell[x][1] == indexᵢ, pvec)
        if !isnothing(start)
            for i = start:length(pcell)
                ind = pvec[i]
                cellᵢ, cellⱼ = pcell[ind]
                if cellᵢ > indexᵢ
                    return nothing
                end
                n = cellpnum[cellᵢ, cellⱼ] += 1
                celllist[cellᵢ, cellⱼ, n] = ind
            end
        end
    end
    return nothing
end
function cellthreadmap_2d!(celllist, cellpnum, pvec, pcell) 
    kernel = @cuda launch=false kernel_cellthreadmap_2d!(celllist, cellpnum, pvec, pcell) 
    config = launch_configuration(kernel.fun)
    threads = min(size(celllist, 1), config.threads)
    blocks = cld(size(celllist, 1), threads)
    CUDA.@sync kernel(celllist, cellpnum, pvec, pcell; threads = threads, blocks = blocks)
end
function fillcells_psort_2d!(celllist, cellpnum, pvec, pcell)
    sortperm!(pvec, pcell; by=first)
    cellthreadmap_2d!(celllist, cellpnum, pvec, pcell) 
end
=#
#####################################################################

function kernel_мaxpairs_2d!(cellpnum, cnt)
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    Nx, Ny = size(cellpnum)
    if  indexᵢ <= Nx && indexⱼ <= Ny 
        n = cellpnum[indexᵢ, indexⱼ] 
        if n > 0
            m         = 0
            neibcellᵢ = indexᵢ - 1
            neibcellⱼ = indexⱼ + 1
            if  0 < neibcellᵢ <= Nx && 0 < neibcellⱼ <= Ny 
                m += cellpnum[neibcellᵢ, neibcellⱼ] 
            end
            neibcellᵢ = indexᵢ 
            neibcellⱼ = indexⱼ + 1
            if 0 < neibcellᵢ <= Nx && 0 < neibcellⱼ <= Ny 
                m += cellpnum[neibcellᵢ, neibcellⱼ] 
            end
            neibcellᵢ = indexᵢ + 1
            neibcellⱼ = indexⱼ + 1
            if 0 < neibcellᵢ <= Nx && 0 < neibcellⱼ <= Ny 
                m += cellpnum[neibcellᵢ, neibcellⱼ] 
            end
            neibcellᵢ = indexᵢ + 1
            neibcellⱼ = indexⱼ 
            if 0 < neibcellᵢ <= Nx && 0 < neibcellⱼ <= Ny 
                m += cellpnum[neibcellᵢ, neibcellⱼ] 
            end
            val  = Int((n * (n - 1)) * 0.5) + m * n
            CUDA.@atomic cnt[1] += val
        end
    end
    return nothing
end
"""
    мaxpairs_2d(cellpnum)

Maximum number of pairs.
"""
function мaxpairs_2d(cellpnum)
    cnt        = CUDA.zeros(Int, 1)
    Nx, Ny     = size(cellpnum)
    gpukernel  = @cuda launch=false kernel_мaxpairs_2d!(cellpnum, cnt)
    config     = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx         = min(maxThreads, Nx)
    Ty         = min(fld(maxThreads, Tx), Ny)
    Bx, By     = cld(Nx, Tx), cld(Ny, Ty) 
    threads    = (Tx, Ty)
    blocks     = (Bx, By)
    CUDA.@sync gpukernel(cellpnum, cnt; threads = threads, blocks = blocks)
    CUDA.@allowscalar cnt[1]
end
#####################################################################

function kernel_neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    #indexₖ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    Nx, Ny = size(cellpnum)
    if indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 1 
        @inbounds len = cellpnum[indexᵢ, indexⱼ]
        for i = 1:len - 1
            @inbounds indi = celllist[i, indexᵢ, indexⱼ]
            for j = i + 1:len
                @inbounds indj = celllist[j, indexᵢ, indexⱼ]
                @inbounds distance = sqrt((points[indi][1] - points[indj][1])^2 + (points[indi][2] - points[indj][2])^2)
                if distance < dist
                    n = CUDA.@atomic cnt[1] += 1
                    if n <= size(pairs, 1)
                        @inbounds  pairs[n + 1] = tuple(indi, indj, distance)
                    end
                end
            end
        end
    end
    return nothing
end
"""
    neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)

Find all pairs with distance < h in one cell.
"""
function neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension ($((CLx, CLy))) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, dist; threads = threads, blocks = blocks)
end
#####################################################################
#=
function kernel_neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, h) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    #indexₖ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if indexᵢ <= size(celllist, 1) && indexⱼ <= size(celllist, 2) && cellpnum[indexᵢ, indexⱼ] > 1 
        @inbounds n   = cellcounter[indexᵢ, indexⱼ]
        @inbounds len = cellpnum[indexᵢ, indexⱼ]
        for i = 1:len - 1
            @inbounds indi = celllist[indexᵢ, indexⱼ, i]
            #if indi > 0
                for j = i + 1:len
                    @inbounds indj = celllist[indexᵢ, indexⱼ, j]
                    #if indj > 0
                        @inbounds distance = sqrt((points[indi][1] - points[indj][1])^2 + (points[indi][2] - points[indj][2])^2)
                        if distance < h
                            n += 1 
                            if n <= size(pairs, 1)
                                @inbounds  pairs[n, indexᵢ, indexⱼ] = tuple(indi, indj, distance)
                            end
                        end
                    #end 
                end
            #end
        end
        cellcounter[indexᵢ, indexⱼ] = n
    end
    return nothing
end
function neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, h)
    kernel = @cuda launch=false kernel_neib_internal_2d!(pairs, cellcounter, cellpnum, celllist, points, h)
    config = launch_configuration(kernel.fun)
    threads = (min(size(celllist, 1), Int(floor(sqrt(config.threads)))), min(size(celllist, 2), Int(floor(sqrt(config.threads)))))
    blocks = (cld(size(celllist, 1), threads[1]), cld(size(celllist, 2), threads[2]))
    CUDA.@sync kernel(pairs, cellcounter, cellpnum, celllist, points, h, pairs; threads = threads, blocks = blocks)
end
=#

function kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist)
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    Nx, Ny = size(cellpnum)
    neibcellᵢ = indexᵢ + offset[1]
    neibcellⱼ = indexⱼ + offset[2]
    if 0 < neibcellᵢ <= Nx &&  0 < neibcellⱼ <= Ny && indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 0 #&& cellpnum[neibcellᵢ, neibcellⱼ] > 0
        iinds = view(celllist, 1:cellpnum[indexᵢ, indexⱼ], indexᵢ, indexⱼ)
        jinds = view(celllist, 1:cellpnum[neibcellᵢ, neibcellⱼ], neibcellᵢ, neibcellⱼ)
        for i in iinds
            for j in jinds
                @inbounds  distance = sqrt((points[i][1] - points[j][1])^2 + (points[i][2] - points[j][2])^2)
                if distance < dist
                    n = CUDA.@atomic cnt[1] += 1
                    if n <= size(pairs, 1)
                        @inbounds pairs[n + 1] = tuple(i, j, distance)
                    end
                end
            end  
        end
    end
    return nothing
end

"""
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)

Find all pairs with another cell shifted on offset.
"""
function neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, offset, dist; threads = threads, blocks = blocks)
end
#####################################################################
#=
function kernel_neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  offset, h)
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    neibcellᵢ = indexᵢ + offset[1]
    neibcellⱼ = indexⱼ + offset[2]
    if 0 < neibcellᵢ <= size(celllist, 1) &&  0 < neibcellⱼ <= size(celllist, 2) && indexᵢ <= size(celllist, 1) && indexⱼ <= size(celllist, 2) && cellpnum[indexᵢ, indexⱼ] > 0 #&& cellpnum[neibcellᵢ, neibcellⱼ] > 0
        n = cellcounter[indexᵢ, indexⱼ]
        iinds = view(celllist, indexᵢ, indexⱼ, 1:cellpnum[indexᵢ, indexⱼ])
        jinds = view(celllist, neibcellᵢ, neibcellⱼ, 1:cellpnum[neibcellᵢ, neibcellⱼ])
        for i in iinds
                for j in jinds
                    @inbounds  distance = sqrt((points[i][1] - points[j][1])^2 + (points[i][2] - points[j][2])^2)
                    if distance < h
                        n += 1
                        if n <= size(pairs, 1)
                            @inbounds pairs[n,  indexᵢ, indexⱼ] = tuple(i, j, distance)
                        end
                    end
                end  
        end
        cellcounter[indexᵢ, indexⱼ] = n
    end
    return nothing
end
function neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  offset, h)
    kernel = @cuda launch=false kernel_neib_external_2d!(pairs, cellcounter, cellpnum, points, celllist,  offset, h)
    config = launch_configuration(kernel.fun)
    threads = (min(size(celllist, 1), Int(floor(sqrt(config.threads)))), min(size(celllist, 2), Int(floor(sqrt(config.threads)))))
    blocks = (cld(size(celllist, 1), threads[1]), cld(size(celllist, 2), threads[2]))
    CUDA.@sync kernel(pairs, cellcounter, cellpnum, points, celllist,  offset, h; threads = threads, blocks = blocks)
end
=#
#####################################################################
# SPH
#####################################################################
function kernel_∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)
            u     = d * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic sumW[pᵢ] += w
            CUDA.@atomic sumW[pⱼ] += w
        end
    end
    return nothing
end
"""

    ∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 

Compute ∑W for each particles pair in list.
"""
function ∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(sumW, pairs, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end

#=
function kernel_∑W_2d!(sumW, cellcounter, pairs, sphkernel, H⁻¹) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    if indexᵢ <= size(cellcounter, 1) &&  indexⱼ <= size(cellcounter, 2) && cellcounter[indexᵢ, indexⱼ] > 0
        for i = 1:cellcounter[indexᵢ, indexⱼ]
            pair  = pairs[i, indexᵢ, indexⱼ]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            u     = d * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic sumW[pᵢ] += w
            CUDA.@atomic sumW[pⱼ] += w
        end
    end
    return nothing
end
"""


∑W_2d!

"""
function ∑W_2d!(sumW, cellcounter, pairs, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑W_2d!(sumW, cellcounter, pairs, sphkernel, H⁻¹) 
    #config = launch_configuration(gpukernel.fun)
    Nx, Ny = size(cellcounter)
    maxThreads = 1024
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(sumW, cellcounter, pairs, sphkernel, H⁻¹; threads = threads, blocks = blocks)
end
=#
#####################################################################

function ∇Wfunc(αD, q, h) 
    if 0 < q < 2
        return αD * 5 * (q - 2) ^ 3 * q / (8h * (q * h + 1e-6)) 
    end
    return 0.0
end
#####################################################################
#=
function kernel_∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    #=
    H   = 1 / H⁻¹
    H⁻² = H⁻¹^2
    C   = 7/π
    αD  = C * H⁻²
    h   = H/2
    h⁻¹ = 1/h
    =#
    if indexᵢ <= size(cellcounter, 1) &&  indexⱼ <= size(cellcounter, 2) && cellcounter[indexᵢ, indexⱼ] > 0
        for i = 1:cellcounter[indexᵢ, indexⱼ]
            pair  = pairs[i, indexᵢ, indexⱼ]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]

            #=
            q = d * h⁻¹
            Wg = ∇Wfunc(αD, q, h)
            ∇w    = ((xᵢ[1] - xⱼ[1]) * Wg, (xᵢ[2] - xⱼ[2]) * Wg)
            =#
            u     = d * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / d
            ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
            
            CUDA.@atomic sum∇W[pᵢ, 1] += ∇w[1]
            CUDA.@atomic sum∇W[pᵢ, 2] += ∇w[2]
            CUDA.@atomic sum∇W[pⱼ, 1] -= ∇w[1]
            CUDA.@atomic sum∇W[pⱼ, 2] -= ∇w[2]
            ∇Wₙ[i, indexᵢ, indexⱼ] = ∇w
        end
    end
    return nothing
end
"""
    
    ∑∇W_2d!

"""
function ∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹) 
    #config = launch_configuration(gpukernel.fun)
    Nx, Ny = size(cellcounter)
    maxThreads = 1024
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹; threads = threads, blocks = blocks)
end
#####################################################################


function kernel_∑∇W_l_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹, cnt) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
            pair  = pairs[index]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            if !isnan(d)
                xᵢ    = points[pᵢ]
                xⱼ    = points[pⱼ]
                u     = d * H⁻¹
                dwk_r = d𝒲(kernel, u, H⁻¹) / d
                ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
            
                CUDA.@atomic sum∇W[pᵢ, 1] += ∇w[1]
                CUDA.@atomic sum∇W[pᵢ, 2] += ∇w[2]
                CUDA.@atomic sum∇W[pⱼ, 1] -= ∇w[1]
                CUDA.@atomic sum∇W[pⱼ, 2] -= ∇w[2]
                n = CUDA.@atomic cnt[1]   += 1
                ∇Wₙ[n + 1] = ∇w
            end
    end
    return nothing
end
function ∑∇W_l_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹) 
    cnt = CUDA.zeros(Int, 1)
    gpukernel = @cuda launch=false kernel_∑∇W_l_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹, cnt) 
    #config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = 1024
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹, cnt; threads = Tx, blocks = Bx)
    #return @allowscalar cnt[1]
end
=#
#####################################################################
function kernel_∑∇W_2d!(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            u     = d * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / d
            ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
            CUDA.@atomic sum∇W[pᵢ, 1] += ∇w[1]
            CUDA.@atomic sum∇W[pᵢ, 2] += ∇w[2]
            CUDA.@atomic sum∇W[pⱼ, 1] -= ∇w[1]
            CUDA.@atomic sum∇W[pⱼ, 2] -= ∇w[2]
            ∇Wₙ[index] = ∇w
        end
    end
    return nothing
end
"""
    
    ∑∇W_2d!(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹) 

Compute gradients.

"""
function ∑∇W_2d!(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑∇W_2d!(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹; threads = Tx, blocks = Bx)
end


#####################################################################

function kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)

            γ⁻¹  = 1/γ
            η²   = (0.1*h)*(0.1*h)
            Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
            DDTgz = ρ₀ * g / Cb
            DDTkh = 2 * h * δᵩ
    
            #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
            =#

            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇Wᵢ   = ∇Wₙ[index]

            #r²    = (xᵢ[1]-xⱼ[1])^2 + (xᵢ[2]-xⱼ[2])^2  #  Δx ⋅ Δx 
            r²    = d^2
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            
            dot3  = -(Δx[1] * ∇Wᵢ[1] + Δx[2] * ∇Wᵢ[2]) #  - Δx ⋅ ∇Wᵢ 
            
            drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀

            visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)

            delta_i    = visc_densi * dot3 * m₀ / ρⱼ

            drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
            visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            delta_j    = visc_densi * dot3 * m₀ / ρᵢ

            m₀dot     = m₀ * (Δv[1] * ∇Wᵢ[1] + Δv[2] * ∇Wᵢ[2])  #  Δv ⋅ ∇Wᵢ

            CUDA.@atomic ∑∂ρ∂t[pᵢ] += (m₀dot + delta_i * MotionLimiter[pᵢ])
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += (m₀dot + delta_j * MotionLimiter[pⱼ])
            
        end
    end
    return nothing
end
"""
    
    ∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 

Compute ∂ρ∂tDDT
"""
function ∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 
    if length(pairs) != length(∇Wₙ) error("Length shoul be equal") end

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter; threads = Tx, blocks = Bx)
end
#####################################################################
function kernel_∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)

            η²    = (0.1 * h) * (0.1 * h)
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            #r²    = (xᵢ[1] - xⱼ[1])^2 + (xᵢ[2] - xⱼ[2])^2 
            r²    = d^2

            ρₘ    = (ρᵢ + ρⱼ) * 0.5
            
            ∇W    = ∇Wₙ[index]

            cond   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            cond_bool = cond < 0

            Δμ   = h * cond / (r² + η²)

            ΔΠ   = cond_bool * (-α * c₀ * Δμ) / ρₘ

            ΔΠm₀∇W = (-ΔΠ * m₀ * ∇W[1], -ΔΠ * m₀ * ∇W[2])

            CUDA.@atomic ∑∂Π∂t[pᵢ, 1] += ΔΠm₀∇W[1]
            CUDA.@atomic ∑∂Π∂t[pᵢ, 2] += ΔΠm₀∇W[2]
            CUDA.@atomic ∑∂Π∂t[pⱼ, 1] -= ΔΠm₀∇W[1]
            CUDA.@atomic ∑∂Π∂t[pⱼ, 2] -= ΔΠm₀∇W[2]

        end
    end
    return nothing
end
"""
    
    ∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀)


Compute ∂Π∂t
"""
function ∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀) 
    gpukernel = @cuda launch=false kernel_∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀; threads = Tx, blocks = Bx)
end
#=
function kernel_∂Π∂t!(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, points, h, ρ, α, v, c₀, m₀) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    if indexᵢ <= size(cellcounter, 1) &&  indexⱼ <= size(cellcounter, 2) && cellcounter[indexᵢ, indexⱼ] > 0

        η²    = (0.1 * h) * (0.1 * h)
    
        for i = 1:cellcounter[indexᵢ, indexⱼ]
            
            pair  = pairs[i, indexᵢ, indexⱼ]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            r²    = xᵢ[1]*xⱼ[1] + xᵢ[2]*xⱼ[2] 

            ρₘ    = (ρᵢ + ρⱼ) * 0.5
            
            ∇W    = ∇Wₙ[i, indexᵢ, indexⱼ]

            cond   = Δv[1]*Δx[1] +  Δv[2]*Δx[2] 

            cond_bool = cond < 0

            Δμ   = h * cond / (r² + η²)

            ΔΠ   = cond_bool * (-α * c₀ * Δμ) / ρₘ

            ΔΠm₀∇W = (-ΔΠ * m₀ * ∇W[1], -ΔΠ * m₀ * ∇W[2])

            CUDA.@atomic ∑∂Π∂t[pᵢ, 1] += ΔΠm₀∇W[1]
            CUDA.@atomic ∑∂Π∂t[pᵢ, 2] += ΔΠm₀∇W[2]
            CUDA.@atomic ∑∂Π∂t[pⱼ, 1] -= ΔΠm₀∇W[1]
            CUDA.@atomic ∑∂Π∂t[pⱼ, 2] -= ΔΠm₀∇W[2]

        end
    end
    return nothing
end
"""
    
    ∂Π∂t!

"""
function ∂Π∂t!(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, points, h, ρ, α, v, c₀, m₀) 
    gpukernel = @cuda launch=false kernel_∂Π∂t!(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, points, h, ρ, α, v, c₀, m₀) 
    config = launch_configuration(gpukernel.fun)
    Nx, Ny = size(cellcounter)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(∑∂Π∂t, ∇Wₙ, cellcounter, pairs, points, h, ρ, α, v, c₀, m₀; threads = threads, blocks = blocks)
end
=#
#####################################################################


"""
    pressure(ρ, c₀, γ, ρ₀)

Pressure
"""
function pressure(ρ, c₀, γ, ρ₀)
    return ((c₀ ^ 2 * ρ₀) / γ) * ((ρ / ρ₀) ^ γ - 1)
end
#####################################################################
function kernel_∂v∂t!(∑∂v∂t,  ∇Wₙ,  pairs, m, ρ, c₀, γ, ρ₀) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)

            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Pᵢ    = pressure(ρᵢ, c₀, γ, ρ₀)
            Pⱼ    = pressure(ρⱼ, c₀, γ, ρ₀)
            ∇W    = ∇Wₙ[index]

            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)

            ∂v∂t  = (- m * Pfac * ∇W[1], - m * Pfac * ∇W[2])

            CUDA.@atomic ∑∂v∂t[pᵢ, 1] +=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂t[pᵢ, 2] +=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂t[pⱼ, 1] -=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂t[pⱼ, 2] -=  ∂v∂t[2]
        end
    end
    return nothing
end
"""
    
    ∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀) 


"""
function ∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀) 
    gpukernel = @cuda launch=false kernel_∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀; threads = Tx, blocks = Bx)
end
#=
function kernel_∂v∂t!(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, points, m, ρ, c₀, γ, ρ₀) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    if indexᵢ <= size(cellcounter, 1) &&  indexⱼ <= size(cellcounter, 2) && cellcounter[indexᵢ, indexⱼ] > 0
    
        for i = 1:cellcounter[indexᵢ, indexⱼ]
            pair  = pairs[i, indexᵢ, indexⱼ]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Pᵢ    = pressure(ρᵢ, c₀, γ, ρ₀)
            Pⱼ    = pressure(ρⱼ, c₀, γ, ρ₀)
            ∇W    = ∇Wₙ[i, indexᵢ, indexⱼ]

            Pfac  = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ∂v∂t  = (- m * Pfac * ∇W[1], - m * Pfac * ∇W[2])

            CUDA.@atomic ∑∂v∂t[pᵢ, 1] +=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂t[pᵢ, 2] +=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂t[pⱼ, 1] -=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂t[pⱼ, 2] -=  ∂v∂t[2]
        end
    end
    return nothing
end
"""
    
    ∂v∂t!


"""
function ∂v∂t!(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, points, m, ρ, c₀, γ, ρ₀) 
    gpukernel = @cuda launch=false kernel_∂v∂t!(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, points, m, ρ, c₀, γ, ρ₀) 
    config = launch_configuration(gpukernel.fun)
    Nx, Ny = size(cellcounter)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    CUDA.@sync gpukernel(∑∂v∂t,  ∇Wₙ, cellcounter, pairs, points, m, ρ, c₀, γ, ρ₀; threads = threads, blocks = blocks)
end
=#
#####################################################################
#dvdtI .= map((x,y) -> x + y * SVector(0, g, 0), dvdtI + viscI, GravityFactor)

function kernel_completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= size(∑∂v∂t, 1)
        ∑∂v∂t[index, 1] +=  ∑∂Π∂t[index, 1] + gvec[1] * gfac[index]
        ∑∂v∂t[index, 2] +=  ∑∂Π∂t[index, 2] + gvec[2] * gfac[index]
    end
    return nothing
end
"""
    
    completed_∂vᵢ∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac)  


"""
function completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac) 
    if size(∑∂v∂t, 1) != size(∑∂Π∂t, 1) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac) 
    config = launch_configuration(gpukernel.fun)
    Nx = size(∑∂v∂t, 1)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, ∑∂Π∂t,  gvec, gfac; threads = Tx, blocks = Bx)
end
#####################################################################

function kernel_update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(ρ)
        ρval = ρ[index] + ∑∂ρ∂t[index] * Δt
        if ρval < ρ₀ && isboundary[index] ρval = ρ₀ end
        ρ[index] = ρval
    end
    return nothing
end
"""
    update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 


"""
function update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 
    if length(ρ) != size(∑∂ρ∂t, 1) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 
    config = launch_configuration(gpukernel.fun)
    Nx = size(∑∂ρ∂t, 1)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary; threads = Tx, blocks = Bx)
end
#####################################################################
function kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= size(∑∂v∂t, 1)
        @inbounds val = v[index]
        @inbounds v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
    end
    return nothing
end
"""
    update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 


"""
function update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    if !(length(v) == size(∑∂v∂t, 1) == length(ml)) error("Wrong length") end
    gpukernel = @cuda launch = false kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = size(∑∂v∂t, 1)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, ∑∂v∂t, Δt, ml; threads = Tx, blocks = Bx)
end

#####################################################################
function kernel_update_xpvΔt!(x, v, Δt, ml) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)
        xval = x[index]
        vval = v[index]
        x[index] = (xval[1] + vval[1] * Δt * ml[index], xval[2] + vval[2] * Δt * ml[index])
    end
    return nothing
end
"""
    update_xpvΔt!(x, v, Δt, ml) 


"""
function update_xpvΔt!(x, v, Δt, ml) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_xpvΔt!(x, v, Δt, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(x, v, Δt, ml; threads = Tx, blocks = Bx)
end
#####################################################################
function kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)
        if ρval < ρ₀ && isboundary[index] ρval = ρ₀ end
        
        ρΔt½[index] = ρ[index] = ρval

        vval = v[index]
        nval = vΔt½[index] = v[index] = (vval[1] + ∑∂v∂t[index, 1] * Δt * ml[index], vval[2] + ∑∂v∂t[index, 2] * Δt * ml[index],)

        xval = x[index]
        xΔt½[index] = x[index] = (xval[1] + (vval[1] + nval[1]) * 0.5  * Δt * ml[index], xval[2] + (vval[2] + nval[2]) * 0.5  * Δt * ml[index])
    
    end
    return nothing
end
"""
    
    update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 


"""
function update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml; threads = Tx, blocks = Bx)
end

#####################################################################

function kernel_Δt_stepping!(buf, v, points, h, η²) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        vp = v[index]
        pp = points[index]
        buf[index] = abs(h * (vp[1] * pp[1] + vp[2] * pp[2]) / (pp[1]^2 + pp[2]^2 + η²))
    end
    return nothing
end
function kernel_Δt_stepping_norm!(buf, a) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        buf[index] =  sqrt(a[index, 1]^2 + a[index, 2]^2) 
    end
    return nothing
end
"""    
    Δt_stepping(buf, a, v, points, c₀, h, CFL) 

"""
function Δt_stepping(buf, a, v, points, c₀, h, CFL) 
    η²  = (0.01)h * (0.01)h

    gpukernel = @cuda launch=false kernel_Δt_stepping_norm!(buf, a) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buf)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(buf, a; threads = Tx, blocks = Bx)

    dt1 = sqrt(h / maximum(buf))

    gpukernel = @cuda launch=false kernel_Δt_stepping!(buf, v, points, h, η²) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buf)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(buf, v, points, h, η²; threads = Tx, blocks = Bx)
    
    visc  = maximum(buf)
    dt2   = h / (c₀ + visc)
    dt    = CFL * min(dt1, dt2)

    return dt
end
