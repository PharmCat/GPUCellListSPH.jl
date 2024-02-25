#####################################################################
# GPU KERNELS FOR 2D
#####################################################################
# CELL LIST
#####################################################################
"""
    cellmap_2d!(pcell, cellpnum, points,  h, offset)  

Map each point to cell and count number of points in each cell.

For each coordinates cell number calculated:

```julia
csᵢ = size(cellpnum, 1) 
p₁  =  (x₁ - offset₁) * h₁⁻¹
pᵢ₁ = ceil(min(max(p₁, 1), csᵢ))
```

"""
function cellmap_2d!(pcell, cellpnum, points,  h, offset)  
    h⁻¹ = (1/h[1], 1/h[2])
    kernel = @cuda launch=false kernel_cellmap_2d!(pcell, cellpnum, points,  h⁻¹, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(size(points, 1), config.threads)
    blocks = cld(size(points, 1), threads)
    CUDA.@sync kernel(pcell, cellpnum, points,  h⁻¹, offset; threads = threads, blocks = blocks)
end
function kernel_cellmap_2d!(pcell, cellpnum, points,  h⁻¹, offset) 
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    csᵢ = size(cellpnum, 1) 
    csⱼ = size(cellpnum, 2) 
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) * h⁻¹[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) * h⁻¹[2]
        pᵢ₁ = ceil(Int32, min(max(p₁, 1), csᵢ)) 
        pᵢ₂ = ceil(Int32, min(max(p₂, 1), csⱼ))
        # maybe add check:  is particle in simulation range? and include only if in simulation area
        pcell[i] = (pᵢ₁, pᵢ₂)

        CUDA.@atomic cellpnum[pᵢ₁, pᵢ₂] += one(Int32) 
    end
    return nothing
end
#####################################################################
"""
    fillcells_naive_2d!(celllist, cellpnum, pcell) 
    
Fill cell list with cell. Naive approach. No bound check. Values in `pcell` list shoid be in range of `cellpnum` and `celllist`.
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
function kernel_fillcells_naive_2d!(celllist, cellpnum, pcell) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if indexᵢ <= length(pcell)
        # no bound check - all should be done before
        pᵢ, pⱼ = pcell[indexᵢ]
        n = CUDA.@atomic cellpnum[pᵢ, pⱼ] += 1
        celllist[n + 1, pᵢ, pⱼ] = indexᵢ
    end
    return nothing
end
#####################################################################
#####################################################################
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
#####################################################################
"""
    neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)

Find all pairs with distance < h in one cell.
"""
function neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension ($((CLx, CLy))) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist², 6)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * Ty * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, dist², cs; threads = threads, blocks = blocks, shmem= Tx * Ty * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist², cs) 
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    #indexₖ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    Nx, Ny = size(cellpnum)
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, blockDim().x * blockDim().y * cs)
    if indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 1
        lind  = indexᵢ + Ny * (indexⱼ - Int32(1))
        ccnt  = zero(Int32)
        len = cellpnum[indexᵢ, indexⱼ]
        for i = 1:len - 1
            indi = celllist[i, indexᵢ, indexⱼ]
            pᵢ = points[indi]
            for j = i + 1:len
                indj = celllist[j, indexᵢ, indexⱼ]
                pⱼ = points[indj]
                distance = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2
                if distance < dist²
                    cache[lind + ccnt] = minmax(indi, indj)
                    ccnt += 1
                    if ccnt == cs
                        ccnt = 0
                        s  = CUDA.@atomic cnt[1] += cs
                        if s + cs <=length(pairs)
                            for cind in 1:cs
                                pairs[s+cind] = cache[lind + cind - 1]
                            end
                        end
                    end
                end
            end
        end
        if ccnt != 0
            s = CUDA.@atomic cnt[1] += ccnt
            if s + ccnt <=length(pairs)
                for cind in 1:ccnt 
                    pairs[s + cind] = cache[lind + cind - 1]
                end
            end
        end
    end
    return nothing
end
#####################################################################
"""
    neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)

Find all pairs with another cell shifted on offset.
"""
function neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², 6)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx) 
    Ty  = min(fld(maxThreads, Tx), Ny)
    Bx, By = cld(Nx, Tx), cld(Ny, Ty)  # Blocks in grid.
    threads = (Tx, Ty)
    blocks  = Bx, By
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * Ty * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, offset, dist², cs; threads = threads, blocks = blocks, shmem= Tx * Ty * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², cs)
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, blockDim().x * blockDim().y * cs)
    Nx, Ny = size(cellpnum)
    neibcellᵢ = indexᵢ + offset[1]
    neibcellⱼ = indexⱼ + offset[2]
    if 0 < neibcellᵢ <= Nx &&  0 < neibcellⱼ <= Ny && indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 0 #&& cellpnum[neibcellᵢ, neibcellⱼ] > 0
        lind  = indexᵢ + Ny * (indexⱼ - Int32(1))
        ccnt  = zero(Int32)
        iinds = view(celllist, 1:cellpnum[indexᵢ, indexⱼ], indexᵢ, indexⱼ)
        jinds = view(celllist, 1:cellpnum[neibcellᵢ, neibcellⱼ], neibcellᵢ, neibcellⱼ)
        for i in iinds
            pᵢ = points[i]
            for j in jinds
                pⱼ = points[j]
                distance = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2
                if distance < dist²
                    cache[lind + ccnt] = minmax(i, j)
                    ccnt += 1
                    if ccnt == cs
                        ccnt = 0
                        s  = CUDA.@atomic cnt[1] += cs
                        if s + cs <=length(pairs)
                            for cind in 1:cs
                                pairs[s+cind] = cache[lind + cind - 1]
                            end
                        end
                    end
                end
            end  
        end
        if ccnt != 0
            s = CUDA.@atomic cnt[1] += ccnt
            if s + ccnt <=length(pairs)
                for cind in 1:ccnt 
                    pairs[s + cind] = cache[lind + cind - 1]
                end
            end
        end
    end
    return nothing
end
#####################################################################
# Make neighbor matrix (list)
#####################################################################
#=
function kernel_neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(points)
        # get point cell
        cell   = pcell[index]
        celli  = cell[1] + offset[1]
        cellj  = cell[2] + offset[2]
        if  0 < celli <= size(celllist, 2) && 0 < cellj <= size(celllist, 3)
            snl    = size(nlist, 1)
            clist  = view(celllist, :, celli, cellj)
            celln  = cellpnum[celli, cellj]
            distsq = dist * dist
            cnt    = ncnt[index]
            pointi = points[index]
            pointj = points[indexj]
            for i = 1:celln
                indexj = clist[i]
                if index != indexj && (pointi[1] - pointj[1])^2 + (pointi[2] - pointj[2])^2 < distsq
                    cnt += 1
                    if cnt <= snl
                        nlist[cnt, index] = indexj
                    end
                end
            end
            ncnt[index] = cnt
        end
    end
    return nothing
end
"""
    neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)

"""
function neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)
    gpukernel = @cuda launch=false kernel_neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)
    config = launch_configuration(gpukernel.fun)
    Nx = length(points)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset; threads = Tx, blocks = Bx)
end
=#
#####################################################################
#####################################################################
# SPH
#####################################################################
"""

    W_2d!(sumW, pairs, sphkernel, H⁻¹) 

Compute W for each particles pair in list.
"""
function W_2d!(W, pairs, points, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_W_2d!(W, pairs, points, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(W, pairs, points, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end
function kernel_W_2d!(W, pairs, points, sphkernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ       = points[pᵢ]
            xⱼ       = points[pⱼ]
            Δx       = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            d        = sqrt(Δx[1]^2 + Δx[2]^2) 
            u        = d * H⁻¹
            w        = 𝒲(sphkernel, u, H⁻¹)
            W[index] = w
        end
    end
    return nothing
end
#####################################################################
#
#####################################################################
"""

    ∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 

Compute ∑W for each particles pair in list.
"""
function ∑W_2d!(∑W, pairs, points, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑W_2d!(∑W, pairs, points, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑W, pairs, points, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end
function kernel_∑W_2d!(∑W, pairs, points, sphkernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            d     = sqrt(Δx[1]^2 + Δx[2]^2) 
            u     = d * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic ∑W[pᵢ] += w
            CUDA.@atomic ∑W[pⱼ] += w
        end
    end
    return nothing
end
#####################################################################
"""
    
    ∇W_2d!(∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Update ∇W.

"""
function ∇W_2d!(∇W, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∇W_2d!(∇W, pairs, points, kernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∇W, pairs, points, kernel, H⁻¹; threads = Tx, blocks = Bx)
end
function kernel_∇W_2d!(∇W, pairs, points, kernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ        = points[pᵢ]
            xⱼ        = points[pⱼ]
            Δx        = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r         = sqrt(Δx[1]^2 + Δx[2]^2) 
            u         = r * H⁻¹
            dwk_r     = d𝒲(kernel, u, H⁻¹) / r
            ∇W[index] = (Δx[1] * dwk_r, Δx[2] * dwk_r)
        end
    end
    return nothing
end
#####################################################################
"""
    
    ∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Update ∑∇W and ∇W.

"""
function ∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∇W, ∇W, pairs, points, kernel, H⁻¹; threads = Tx, blocks = Bx)
end
function kernel_∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r     = sqrt(Δx[1]^2 + Δx[2]^2) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇w    = (Δx[1] * dwk_r, Δx[2] * dwk_r)
            if isnan(dwk_r) 
                @cuprintln "kernel W_2d  dwk_r = $dwk_r, pair = $pair"
                error() 
            end
            ∑∇Wˣ = ∑∇W[1]
            ∑∇Wʸ = ∑∇W[2]
            CUDA.@atomic ∑∇Wˣ[pᵢ] += ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pᵢ] += ∇w[2]
            CUDA.@atomic ∑∇Wˣ[pⱼ] -= ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pⱼ] -= ∇w[2]
            ∇W[index] = ∇w
        end
    end
    return nothing
end
#####################################################################
function kernel_∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r     = sqrt(Δx[1]^2 + Δx[2]^2) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇w    = (Δx[1] * dwk_r, Δx[2] * dwk_r)
            ∑∇Wˣ = ∑∇W[1]
            ∑∇Wʸ = ∑∇W[2]
            CUDA.@atomic ∑∇Wˣ[pᵢ] += ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pᵢ] += ∇w[2]

            CUDA.@atomic ∑∇Wˣ[pⱼ] -= ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pⱼ] -= ∇w[2]
        end
    end
    return nothing
end
"""
    
    ∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Update ∑∇W.

"""
function ∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∇W, pairs, points, kernel, H⁻¹; threads = Tx, blocks = Bx)
end


#####################################################################
"""
    
    ∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 

Compute ∂ρ∂t - density derivative includind density diffusion.



"""
function ∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    if length(pairs) != length(∇W) error("Length shoul be equal") end

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    tindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    index      = tindex
    # move it outside kernel
    γ⁻¹  = 1/γ
    η²   = (0.1*h)*(0.1*h)
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb
    DDTkh = 2 * h * δᵩ

    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ > 0 # && !(isboundary[pᵢ] && isboundary[pᵢ]) 
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
            #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
            =#
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇Wᵢⱼ  = ∇W[index]
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            dot3  = -(Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2]) #  - Δx ⋅ ∇Wᵢⱼ

            # as actual range at timestep Δt½  may be greateg  - some problems can be here
            if 1 + DDTgz * Δx[2] < 0 || 1 - DDTgz * Δx[2] < 0 return nothing end
            
            m₀dot     = m₀ * (Δv[1] * ∇Wᵢⱼ[1] + Δv[2] * ∇Wᵢⱼ[2])  #  Δv ⋅ ∇Wᵢⱼ
            ∑∂ρ∂ti = ∑∂ρ∂tj = m₀dot

            if ptype[pᵢ] >= 1
                drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀ ## << CHECK
                visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
                delta_i    = visc_densi * dot3 * m₀ / ρⱼ
                ∑∂ρ∂ti    += delta_i 
            end
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti 

            if ptype[pⱼ] >= 1
                drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
                visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
                delta_j    = visc_densi * dot3 * m₀ / ρᵢ
                ∑∂ρ∂tj    += delta_j 
            end
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += ∑∂ρ∂tj
            
            #=
            if isnan(delta_j) || isnan(m₀dot)  || isnan(ρᵢ) || isnan(ρⱼ) 
                @cuprintln "kernel_DDT 1 isnan dx1 = $(Δx[1]) , dx2 = $(Δx[2]) rhoi = $ρᵢ , dot3 = $dot3 , visc_densi = $visc_densi drhopvn = $drhopvn $(∇W[1]) $(Δv[1])"
                error() 
            end
            if isinf(delta_j) || isinf(m₀dot)  || isinf(delta_i) 
                @cuprintln "kernel_DDT 2 inf: dx1 = $(Δx[1]) , dx2 = $(Δx[2]) rhoi = $ρᵢ , rhoj = $ρⱼ , dot3 = $dot3 ,  delta_i = $delta_i , delta_j = $delta_j , drhopvn = $drhopvn , visc_densi = $visc_densi , $(∇W[1]) , $(Δv[1])"
                error() 
            end
            =#
            #mlfac = MotionLimiter[pᵢ] * MotionLimiter[pⱼ]
            #=
            if isnan(∑∂ρ∂tval1) || isnan(∑∂ρ∂tval2) || abs(∑∂ρ∂tval1) >  10000000 || abs(∑∂ρ∂tval2) >  10000000
                @cuprintln "kernel DDT: drhodti = $∑∂ρ∂ti drhodtj = $∑∂ρ∂tj, dx1 = $(Δx[1]), dx2 = $(Δx[2]) rhoi = $ρᵢ, rhoj = $ρⱼ, dot3 = $dot3, visc_densi = $visc_densi, drhopvn = $drhopvn, dw = $(∇W[1]),  dv = $(Δv[1])"
                error() 
            end
            =#
            
        end
        index += stride
    end
    return nothing
end
#####################################################################
"""
    
    ∂Π∂t!(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀)


Compute ∂Π∂t - artificial viscosity.
"""
function ∂Π∂t!(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀) 
    gpukernel = @cuda launch=false kernel_∂Π∂t!(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀; threads = Tx, blocks = Bx)
end
function kernel_∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0

            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
            η²    = (0.1 * h) * (0.1 * h)
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            #=
            if isnan(ρᵢ) || iszero(ρᵢ) || ρᵢ < 0.001 || isnan(ρⱼ) || iszero(ρⱼ) || ρⱼ < 0.001
                @cuprintln "kernel Π: index =  $index, rhoi = $ρᵢ, rhoi = $ρⱼ, dx = $Δx, r =  $r², pair = $pair"
                error() 
            end
            =#
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            ρₘ    = (ρᵢ + ρⱼ) * 0.5
            ∇W    = ∇Wₙ[index]
            cond   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            if cond < 0
                Δμ   = h * cond / (r² + η²)
                ΔΠ   =  (-α * c₀ * Δμ) / ρₘ
                ΔΠm₀∇W = (-ΔΠ * m₀ * ∇W[1], -ΔΠ * m₀ * ∇W[2])
                
                if isnan(ΔΠm₀∇W[1])
                    @cuprintln "kernel Π: Π = $ΔΠ ,  W = $(∇W[1])"
                    error() 
                end
                #CUDA.@atomic ∑∂Π∂t[pᵢ, 1] += ΔΠm₀∇W[1]
                #CUDA.@atomic ∑∂Π∂t[pᵢ, 2] += ΔΠm₀∇W[2]
                #CUDA.@atomic ∑∂Π∂t[pⱼ, 1] -= ΔΠm₀∇W[1]
                #CUDA.@atomic ∑∂Π∂t[pⱼ, 2] -= ΔΠm₀∇W[2]
                ∑∂Π∂tˣ = ∑∂Π∂t[1]
                ∑∂Π∂tʸ = ∑∂Π∂t[2]   
                CUDA.@atomic ∑∂Π∂tˣ[pᵢ] += ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂Π∂tʸ[pᵢ] += ΔΠm₀∇W[2]
                CUDA.@atomic ∑∂Π∂tˣ[pⱼ] -= ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂Π∂tʸ[pⱼ] -= ΔΠm₀∇W[2]
            end
        end
    end
    return nothing
end
#####################################################################
"""
    pressure(ρ, c₀, γ, ρ₀)

Equation of State in Weakly-Compressible SPH
"""
#=
function pressure(ρ, c₀, γ, ρ₀)
    return ((c₀ ^ 2 * ρ₀) / γ) * ((ρ / ρ₀) ^ γ - 1)
end
function pressure(ρ, c₀, γ, ρ₀, γ⁻¹::Float64)
    return (c₀ ^ 2 * ρ₀ * γ⁻¹) * ((ρ / ρ₀) ^ γ - 1)
end
=#
# The correction is to be applied on boundary particles
# J. P. Hughes and D. I. Graham, “Comparison of incompressible and weakly-compressible SPH models for free-surface water flows”, Journal of Hydraulic Research, 48 (2010), pp. 105-117.
function pressure(ρ, γ, ρ₀, P₀, ptype)
    if ptype < 1 && ρ < ρ₀
        return 0.0
    end
    return  P₀ * ((ρ / ρ₀) ^ γ - 1)
end

#####################################################################
"""
    
    pressure!(P, ρ, ρ₀, c₀, γ) 

Equation of State in Weakly-Compressible SPH.



"""
function pressure!(P, ρ, c₀, γ, ρ₀, ptype) 
    if length(P) != length(ρ) != length(ptype) error("Wrong length") end
    P₀  =  c₀ ^ 2 * ρ₀ / γ
    gpukernel = @cuda launch=false kernel_pressure!(P, ρ, γ, ρ₀, P₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(ρ)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(P, ρ, γ, ρ₀, P₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_pressure!(P, ρ, γ, ρ₀, P₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(ρ)
        P[index] = pressure(ρ[index], γ, ρ₀, P₀, ptype[index])
    end
    return nothing
end
#####################################################################
"""
    
    ∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀) 

The momentum equation (without dissipation).
"""
function ∂v∂t!(∑∂v∂t,  ∇W, P, pairs, m, ρ) 
    gpukernel = @cuda launch=false kernel_∂v∂t!(∑∂v∂t,  ∇W, P, pairs, m, ρ) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t,  ∇W, P, pairs, m, ρ; threads = Tx, blocks = Bx)
end
function kernel_∂v∂t!(∑∂v∂t, ∇Wₙ, P, pairs, m, ρ) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0

            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            
            Pᵢ    = P[pᵢ]
            Pⱼ    = P[pⱼ]
            ∇W    = ∇Wₙ[index]

            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)

            ∂v∂t  = (- m * Pfac * ∇W[1], - m * Pfac * ∇W[2])
            
            #=
            if isnan(∂v∂t[1])
                @cuprintln "kernel dvdt: rhoi = $ρᵢ , Pi =  $Pᵢ , m = $m , Pfac = $Pfac , W1 = $(∇W[1])"
                error() 
            end
            if isnan(ρᵢ) || iszero(ρᵢ) || ρᵢ < 0.001 || isnan(ρⱼ) || iszero(ρⱼ) || ρⱼ < 0.001
                @cuprintln "kernel update rho: index =  $index , rhoi = $ρᵢ , rhoi = $ρⱼ, dpdt =  $(∑∂v∂t[index]), pair = $pair"
                error() 
            end
            =#
            #CUDA.@atomic ∑∂v∂t[pᵢ, 1] +=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂t[pᵢ, 2] +=  ∂v∂t[2]
            #CUDA.@atomic ∑∂v∂t[pⱼ, 1] -=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂t[pⱼ, 2] -=  ∂v∂t[2]
            ∑∂v∂tˣ = ∑∂v∂t[1]
            ∑∂v∂tʸ = ∑∂v∂t[2]   
            CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  ∂v∂t[2]
            
        end
    end
    return nothing
end
#####################################################################

"""
    
    completed_∂vᵢ∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac)  

Add gravity and artificial viscosity to the momentum equation.
"""
function completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec) 
    if length(∑∂v∂t[1]) != length(∑∂Π∂t[1]) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(∑∂v∂t[1])
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, ∑∂Π∂t,  gvec; threads = Tx, blocks = Bx)
end
function kernel_completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∑∂v∂t[1])
        #∑∂v∂t[index, 1] +=  ∑∂Π∂t[index, 1] - gvec[1] #* gfac[index]
        #∑∂v∂t[index, 2] +=  ∑∂Π∂t[index, 2] - gvec[2] #* gfac[index]
        ∑∂v∂tˣ = ∑∂v∂t[1]
        ∑∂v∂tʸ = ∑∂v∂t[2]
        ∑∂Π∂tˣ = ∑∂Π∂t[1]
        ∑∂Π∂tʸ = ∑∂Π∂t[2] 
        ∑∂v∂tˣ[index] +=  ∑∂Π∂tˣ[index] - gvec[1] #* gfac[index]
        ∑∂v∂tʸ[index] +=  ∑∂Π∂tʸ[index] - gvec[2] #* gfac[index]
        
    end
    return nothing
end
#####################################################################
"""
    update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 


"""
function update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype) 
    if length(ρ) != length(∑∂ρ∂t) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(ρ)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(ρ)
        ρval = ρ[index] + ∑∂ρ∂t[index] * Δt
        if ρval < ρ₀ && ptype[index] < 1 ρval = ρ₀ end
        #=
        if isnan(ρval) || iszero(ρval) || ρval < 0.001
            @cuprintln "kernel update rho: index =  $index, rhoval = $ρval, rhoi = $(ρ[index]), dpdt = $(∑∂ρ∂t[index]), dt = $Δt, isboundary = $(isboundary[index])"
            error() 
        end
        =#
        ρ[index] = ρval
    end
    return nothing
end
#####################################################################
"""
    update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 


"""
function update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 
    if !(length(v) == length(∑∂v∂t[1]) == length(ptype)) error("Wrong length") end
    gpukernel = @cuda launch = false kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(v)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, ∑∂v∂t, Δt, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(v) && ptype[index] >= 1
        val = v[index]
        #v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        ∑∂v∂tˣ = ∑∂v∂t[1]
        ∑∂v∂tʸ = ∑∂v∂t[2] 
        v[index] = (val[1] + ∑∂v∂tˣ[index] * Δt, val[2] + ∑∂v∂tʸ[index] * Δt)
    
        #=
        if isnan(v[index][1] )
            @cuprintln "kernel update v by dvdvt: val = $(val[1]) , dvdt =  $(∑∂v∂t[index, 1] ), dt =  $Δt"
            error() 
        end
        =#
    end
    return nothing
end
#####################################################################
"""
    update_xpvΔt!(x, v, Δt, ml) 


"""
function update_xpvΔt!(x, v, Δt) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_xpvΔt!(x, v, Δt) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(x, v, Δt; threads = Tx, blocks = Bx)
end
function kernel_update_xpvΔt!(x, v, Δt) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)
        xval = x[index]
        vval = v[index]
        x[index] = (xval[1] + vval[1] * Δt, xval[2] + vval[2] * Δt)
        #=
        if isnan(x[index][1] )
            @cuprintln "kernel dxdt: xval =  $(xval[1]) , vval =  $(vval[1]),  dt = $Δt"
            error() 
        end
        =#
    end
    return nothing
end
#####################################################################
"""
    
    update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 


"""
function update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) # << rename
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)
        if ρval < ρ₀ && ptype[index] < 1 ρval = ρ₀ end

        #=
        if isnan(ρval) || iszero(ρval) || ρval < 0.01
            @cuprintln "kernel update all rho: rhova = $ρval, epsi = $epsi, drhodt = $(∑∂ρ∂t[index]), rhot12 = $(ρΔt½[index]), dt = $Δt"
            error() 
        end
        =#
        ρΔt½[index] = ρval
        ρ[index]    = ρval
        #=
        if ρΔt½[index] < 0.01
            @cuprintln "kernel update all rho 1: rhova = $ρval , epsi = $epsi , drhodt = $(∑∂ρ∂t[index]) , rhot12 = $(ρΔt½[index]) $Δt"
            error() 
        end
        if ρ[index]  < 0.01
            @cuprintln "kernel update all rho 1: rhova = $ρval , epsi = $epsi , drhodt = $(∑∂ρ∂t[index]) , rhot12 = $(ρΔt½[index]) $Δt"
            error() 
        end
        =#
        vval        = v[index]
        ∑∂v∂tˣ      = ∑∂v∂t[1]
        ∑∂v∂tʸ      = ∑∂v∂t[2] 
        ml          = ifelse(ptype[index] >= 1, 1.0, 0.0)
        nval        = (vval[1] +  ∑∂v∂tˣ[index] * Δt * ml, vval[2]  + ∑∂v∂tʸ[index] * Δt * ml)
        vΔt½[index] = nval
        v[index]    = nval

        xval           = x[index]
        Δxˣ, Δxʸ       = (vval[1] + nval[1]) * 0.5  * Δt, (vval[2] + nval[2]) * 0.5  * Δt
        cΔx[1][index] += Δxˣ
        cΔx[2][index] += Δxʸ
        xval           = (xval[1] + Δxˣ, xval[2] + Δxʸ)
        xΔt½[index]    = xval
        x[index]       = xval
    end
    return nothing
end
#####################################################################
"""    
    Δt_stepping(buf, a, v, points, c₀, h, CFL, timelims) 

"""
function Δt_stepping(buf, a, v, points, c₀, h, CFL, timelims) 

    # some problems can be here if we have cells with big acceleration 
    # may be include only particles that only in simulation range

    η²  = (0.01)h * (0.01)h

    gpukernel = @cuda launch=false kernel_Δt_stepping_norm!(buf, a) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buf)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(buf, a; threads = Tx, blocks = Bx)

    dt1 = sqrt(h / 3maximum(buf)) # mul 1/3

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
    dt    = min(max(dt, timelims[1]), timelims[2])
    return dt
end
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
        buf[index] =  sqrt(a[1][index]^2 + a[2][index]^2) 
    end
    return nothing
end
#####################################################################
#####################################################################
"""
    
    ∂v∂tpF!(∑∂v∂t, pairs, points, s, H) 

Add surface tension to ∑∂v∂t. Modified.

A. Tartakovsky and P. Meakin, Phys. Rev. E 72 (2005)
"""
function ∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, ptype) 
    gpukernel = @cuda launch=false kernel_∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, pairs, points, s, h, m₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            if ptype[pᵢ] >= 1 && ptype[pⱼ] >= 1
                xᵢ    = points[pᵢ]
                xⱼ    = points[pⱼ]
                Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
                r     = sqrt(Δx[1]^2 + Δx[2]^2) 
                if r < 2h
                    scos = s * cos(1.5π * r / 2h)/ (r + (0.1*h))
                    ∑∂v∂tˣ = ∑∂v∂t[1]
                    ∑∂v∂tʸ = ∑∂v∂t[2] 
                    CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  scos * Δx[1] / m₀
                    CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  scos * Δx[2] / m₀
                    CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  scos * Δx[1] / m₀
                    CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  scos * Δx[2] / m₀
                end
            end
        end
    end
    return nothing
end
###################################################################################
# Dynamic Particle Collision (DPC) 
# https://arxiv.org/pdf/2110.10076.pdf
# Stability and accuracy of the weakly compressible SPH with par-
# ticle regularization techniques
# Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia
###################################################################################
"""
    
    dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim) 

Dynamic Particle Collision (DPC) correction.


```math
\\delta \\textbf{v}_i^{DPC} = \\sum k_{ij}\\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \\frac{\\Delta  t}{\\rho_i}\\sum \\phi_{ij} \\frac{2V_j}{V_i + V_j}\\frac{p_{ij}^b}{r_{ij}^2 + \\eta^2}\\textbf{r}_{ij}

\\\\

(v_{ij}^{coll} , \\quad \\phi_{ij}) = \\begin{cases} (\\frac{\\textbf{v}_{ij}\\cdot \\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}\textbf{r}_{ji}, \\quad 0) & \\textbf{v}_{ij}\\cdot \\textbf{r}_{ij} < 0 \\\\ (0, \\quad 1) &  otherwise \\end{cases}

\\\\
p_{ij}^b = \\tilde{p}_{ij} \\chi_{ij} 

\\\\

\\tilde{p}_{ij} = max(min(\\lambda |p_i + p_j|, \\lambda p_{max}), p_{min})

\\\\

\\chi_{ij}  = \\sqrt{\\frac{\\omega({r}_{ij}, l_0)}{\\omega(l_0/2, l_0)}}

\\\\

k_{ij} =  \\begin{cases} \\chi_{ij} & 0.5 \\le {r}_{ij}/l_0 < 1 \\\\ 1 & {r}_{ij}/l_0 < 0.5 \\end{cases}

```

Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf

"""
function dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim)
    l₀⁻¹     = 1 / l₀  
    wh⁻¹     = 1 / 𝒲(sphkernel, 0.5, l₀⁻¹)
    gpukernel = @cuda launch=false kernel_dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim; threads = Tx, blocks = Bx)
end
function kernel_dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            η²    = (0.1 * l₀) * (0.1 * l₀)
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            r²    = Δx[1]^2 + Δx[2]^2 
            r     = sqrt(r²) 
            u     = r * l₀⁻¹
            w     = 𝒲(sphkernel, u, l₀⁻¹)

            χ     = sqrt(w * wh⁻¹)

            k     = ifelse(u < dpckernlim, 1.0, χ)

            Pᵇ    = χ * max(min(λ * abs(P[pᵢ] + P[pⱼ]), λ * Pmax), Pmin)

            vr   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            if vr < 0
                # Δvdpc = ∑ k * 2mⱼ / (mᵢ + mⱼ) * vᶜ   | mⱼ = mᵢ |  => Δvdpc = ∑ k * vᶜ
                vrdr    = vr / (r² + η²)
                vᶜ      = (-vrdr * Δx[1],  -vrdr * Δx[2])
                Δvdpc   = (k * vᶜ[1],  k * vᶜ[2])
            else
                # Δvdpc = Δt / ρᵢ * ∑ 2Vᵢ / (Vᵢ + Vⱼ) * Pᵇ / (r² + η²) * Δx
                # V = m / ρ
                # Δvdpc = Δt * ∑ 2 / (ρᵢ + ρⱼ) * Pᵇ / (r² + η²) * Δx
                tvar = 2Δt* Pᵇ / ((ρᵢ + ρⱼ) * (r² + η²))
                Δvdpc = (tvar * Δx[1], tvar * Δx[2])
            end
            
            ∑Δvdpcˣ = ∑Δvdpc[1]
            ∑Δvdpcʸ = ∑Δvdpc[2]   
            CUDA.@atomic ∑Δvdpcˣ[pᵢ] +=  Δvdpc[1]
            CUDA.@atomic ∑Δvdpcʸ[pᵢ] +=  Δvdpc[2]
            CUDA.@atomic ∑Δvdpcˣ[pⱼ] -=  Δvdpc[1]
            CUDA.@atomic ∑Δvdpcʸ[pⱼ] -=  Δvdpc[2]
        end
    end
    return nothing
end

"""
    update_dpcreg!(v, x, ∑Δvdpc, Δt, isboundary) 

Update velocity and position.
"""
function update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, x, ∑Δvdpc, Δt, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)
        if ptype[index] >= 1
            xval = x[index]
            vval = v[index]
            dpcval = (∑Δvdpc[1][index], ∑Δvdpc[2][index])

            v[index] = (vval[1] + dpcval[1], vval[2] + dpcval[2])
            x[index] = (xval[1] + dpcval[1] * Δt, xval[2] + dpcval[2] * Δt)
        end
    end
    return nothing
end
###################################################################################
# Corrected Smoothed Particle Method (CSPM)
# Chen, J. K., Beraun, J. E., & Carney, T. C. (1999). 
# A corrective smoothed particle method for boundary value problems in heat conduction. International Journal for Numerical Methods in Engineering, 
# 46(2), 231–252. doi:10.1002/(sici)1097-0207(19990920)46:2<231::aid-nme672>3.0.co;2-k
# https://upcommons.upc.edu/bitstream/handle/2117/187607/Particles_2017-82_A%20SPH%20model%20for%20prediction.pdf
# A SPH MODEL FOR PREDICTION OF OIL SLICK DIAMETER IN
# THE GRAVITY-INERTIAL SPREADING PHASE
###################################################################################
"""
    
    cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹)


"""
function cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹)
    gpukernel = @cuda launch=false kernel_cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end
function kernel_cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            η²    = (0.1 * l₀) * (0.1 * l₀)
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            r     = sqrt(r²) 
            u     = r * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)

            CUDA.@atomic ∑ρcspm1[pᵢ] +=  m₀ * w
            CUDA.@atomic ∑ρcspm2[pᵢ] +=  w * m₀ / ρⱼ

            CUDA.@atomic ∑ρcspm1[pⱼ] +=  m₀ * w
            CUDA.@atomic ∑ρcspm2[pⱼ] +=  w * m₀ / ρᵢ
        end
    end
    return nothing
end
#####################################################################
# XSPH Correction 
#####################################################################
"""
    
    xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀)


"""
function xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀)
    gpukernel = @cuda launch=false kernel_xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑Δvxsph, v, ρ, W, pairs, m₀; threads = Tx, blocks = Bx)
end
function kernel_xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            
            xsph  = 2m₀*W[index] / (ρᵢ + ρⱼ)
            xsphv = (xsph * Δv[1], xsph * Δv[2])

            ∑Δvxsphˣ = ∑Δvxsph[1]
            ∑Δvxsphʸ = ∑Δvxsph[2]
            CUDA.@atomic ∑Δvxsphˣ[pᵢ] -=  xsphv[1]
            CUDA.@atomic ∑Δvxsphʸ[pᵢ] -=  xsphv[2]
            CUDA.@atomic ∑Δvxsphˣ[pⱼ] +=  xsphv[1]
            CUDA.@atomic ∑Δvxsphʸ[pⱼ] +=  xsphv[2]
        end
    end
    return nothing
end