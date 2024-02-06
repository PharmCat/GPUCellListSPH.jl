#####################################################################
# CELL LIST
#####################################################################

"""
    Map each point to cell.
"""
function kernel_cellmap_2d!(pcell, points,  h, offset) 
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) / h[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) / h[2]
        pᵢ₁ = ceil(Int32, p₁)
        pᵢ₂ = ceil(Int32, p₂)
        @inbounds pcell[i] = (pᵢ₁, pᵢ₂) # what to with points outside cell grid?
    end
    return nothing
end
function cellmap_2d!(pcell, points, h, offset) 
    kernel = @cuda launch=false kernel_cellmap_2d!(pcell, points,  h, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(size(points, 1), config.threads)
    blocks = cld(size(points, 1), threads)
    CUDA.@sync kernel(pcell, points,  h, offset; threads = threads, blocks = blocks)
end


"""
    Number of points in each cell.
"""
function kernel_cellpnum_2d!(cellpnum, points,  h, offset) 
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    csᵢ = size(cellpnum, 1) 
    csⱼ = size(cellpnum, 2) 
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) / h[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) / h[2]
        pᵢ₁ = ceil(Int32, p₁) 
        pᵢ₂ = ceil(Int32, p₂)
        if csᵢ >= pᵢ₁ > 0 && csⱼ >= pᵢ₂ > 0
            CUDA.@atomic cellpnum[pᵢ₁, pᵢ₂] += one(Int32) 
        end
    end
    return nothing
end
function cellpnum_2d!(cellpnum, points,  h, offset)  
    kernel = @cuda launch=false kernel_cellpnum_2d!(cellpnum, points,  h, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(size(points, 1), config.threads)
    blocks = cld(size(points, 1), threads)
    CUDA.@sync kernel(cellpnum, points,  h, offset; threads = threads, blocks = blocks)
end

"""
    Fill cell list with cell. Naive approach.
"""
function kernel_fillcells_naive_2d!(celllist, cellpnum, pcell) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if indexᵢ <= length(pcell) 
        pᵢ, pⱼ = pcell[indexᵢ]
        n = CUDA.@atomic cellpnum[pᵢ, pⱼ] += 1
        celllist[pᵢ, pⱼ, n + 1] = indexᵢ
    end
    return nothing
end
function fillcells_naive_2d!(celllist, cellpnum, pcell)  
    kernel = @cuda launch=false kernel_fillcells_naive_2d!(celllist, cellpnum, pcell) 
    config = launch_configuration(kernel.fun)
    threads = min(length(pcell), config.threads)
    blocks = cld(length(pcell), threads)
    CUDA.@sync kernel(celllist, cellpnum, pcell; threads = threads, blocks = blocks)
end


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

"""
    Find all pairs with distance < h
"""
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

"""
    Find all pairs with another cell shifted on offset.
"""
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

#####################################################################
# SPH
#####################################################################

"""
    ∑W_2d!


"""
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

"""
    ∑∇W_2d!


"""
function kernel_∑∇W_2d!(sum∇W, ∇Wₙ, cellcounter, pairs, points, kernel, H⁻¹) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    if indexᵢ <= size(cellcounter, 1) &&  indexⱼ <= size(cellcounter, 2) && cellcounter[indexᵢ, indexⱼ] > 0
        for i = 1:cellcounter[indexᵢ, indexⱼ]
            pair  = pairs[i, indexᵢ, indexⱼ]
            pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            u     = d * H⁻¹

            dwk_r = d𝒲(kernel, u, H⁻¹) / d

            ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)

            sum∇W[pᵢ, 1] += ∇w[1]
            sum∇W[pᵢ, 2] += ∇w[2]
            CUDA.@atomic sum∇W[pⱼ, 1] -= ∇w[1]
            CUDA.@atomic sum∇W[pⱼ, 2] -= ∇w[2]
            ∇Wₙ[i, indexᵢ, indexⱼ] = ∇w
        end
    end
    return nothing
end
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

#=

        q = d / h

        Wg = Optim∇ᵢWᵢⱼ(αD, q, xᵢⱼ[iter], h)

        sumWgI[i] +=  Wg
        sumWgI[j] -=  Wg

        sumWgL[iter] = Wg


maxThreads = 1024
        Nx, Ny, Nz = size(f)
        Tx  = min(maxThreads, Nx)
        Ty  = min(fld(maxThreads, Tx), Ny)
        Tz  = min(fld(maxThreads, (Tx*Ty)), Nz)

        Bx, By, Bz = cld(Nx, Tx), cld(Ny, Ty), cld(Nz, Tz)  # Blocks in grid.
=#