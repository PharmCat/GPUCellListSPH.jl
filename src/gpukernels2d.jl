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
function kernel_neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist², cs)  # cs - cache length for each thread
    threadx = threadIdx().x
    thready = threadIdx().y
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadx
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + thready
    #indexₖ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z 
    Nx, Ny = size(cellpnum)
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x , blockDim().y))
    if indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 1
        ccnt  = zero(Int32)
        len = cellpnum[indexᵢ, indexⱼ]
        for i = 1:len - 1
            indi = celllist[i, indexᵢ, indexⱼ]
            pᵢ  = points[indi]
            for j = i + 1:len
                indj = celllist[j, indexᵢ, indexⱼ]
                pⱼ   = points[indj]
                distance = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2 # calculate r² to awoid sqrt, compare with squared maximum distance dist²
                if distance < dist²
                    #n = CUDA.@atomic cnt[1] += 1
                    #pairs[n+1] = minmax(indi, indj)
                    ccnt += 1
                    cache[ccnt, threadx, thready] = minmax(indi, indj)
                    if ccnt == cs
                        s  = CUDA.@atomic cnt[1] += ccnt
                        if s + ccnt <=length(pairs)
                            for cind in 1:ccnt
                                pairs[s + cind] = cache[cind, threadx, thready]
                            end
                        end
                        ccnt = 0
                    end
                end
            end
        end
        if ccnt > 0 
            s  = CUDA.@atomic cnt[1] += ccnt
            if s + ccnt <=length(pairs)
                for cind in 1:ccnt
                    pairs[s + cind] = cache[cind, threadx, thready]
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
    threadx = threadIdx().x
    thready = threadIdx().y
    indexᵢ = (blockIdx().x - Int32(1)) * blockDim().x + threadx
    indexⱼ = (blockIdx().y - Int32(1)) * blockDim().y + thready
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x , blockDim().y))
    Nx, Ny = size(cellpnum)
    neibcellᵢ = indexᵢ + offset[1]
    neibcellⱼ = indexⱼ + offset[2]
    if 0 < neibcellᵢ <= Nx &&  0 < neibcellⱼ <= Ny && indexᵢ <= Nx && indexⱼ <= Ny && cellpnum[indexᵢ, indexⱼ] > 0 #&& cellpnum[neibcellᵢ, neibcellⱼ] > 0
        ccnt  = zero(Int32)
        iinds = view(celllist, 1:cellpnum[indexᵢ, indexⱼ], indexᵢ, indexⱼ)
        jinds = view(celllist, 1:cellpnum[neibcellᵢ, neibcellⱼ], neibcellᵢ, neibcellⱼ)
        for i in iinds
            pᵢ = points[i]
            for j in jinds
                pⱼ = points[j]
                distance = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2
                if distance < dist²
                    #n = CUDA.@atomic cnt[1] += 1
                    #pairs[n + 1] = minmax(i, j)
                    
                    ccnt += 1
                    cache[ccnt, threadx, thready] = minmax(i, j)
                    if ccnt == cs
                        s  = CUDA.@atomic cnt[1] += ccnt
                        if s + ccnt <=length(pairs)
                            for cind in 1:ccnt
                                pairs[s + cind] = cache[cind, threadx, thready]
                            end
                        end
                        ccnt = 0
                    end 

                end
            end  
        end   
        if ccnt > 0 
            s  = CUDA.@atomic cnt[1] += ccnt
            if s + ccnt <=length(pairs)
                for cind in 1:ccnt
                    pairs[s + cind] = cache[cind, threadx, thready]
                end
            end
        end
        
    end
    return nothing
end

#####################################################################
# 
#####################################################################
function pranges!(ranges, pairs) 
    gpukernel = @cuda launch=false kernel_pranges!(ranges, pairs) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(ranges)
    maxThreads = config.threads
    Tx  = min(32, maxThreads, Nx)
    CUDA.@sync gpukernel(ranges, pairs; threads = Tx, blocks = 1, shmem = Tx * sizeof(Int))
end
function kernel_pranges!(ranges, pairs) 
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    thread = threadIdx().x
    stride = gridDim().x * blockDim().x
    tn     = blockDim().x
    cache  = CuDynamicSharedArray(Int, tn)
    cache[thread] = 1
    rs = 0
    re = 0
    sync_threads()
    while index <= length(ranges)
        s = false
        rs = 0
        re = 0
        for i = cache[thread]:length(pairs)
            fpi = first(pairs[i])
            if fpi == index && !s
                s  = true
                rs = i
            elseif fpi != index && s
                re = i - 1
                ranges[index] = (rs, re)
                cache[thread] = i
                break
            #elseif (!s && index < fpi) || fpi == 0
            #    ranges[index] = (0, 0)
            #    break
            end
        end
        if s  == true && re == 0 
            ranges[index] = (rs, length(pairs))
        end
        sync_threads()
        if thread == 1
            maxindex = 1
            for i = 1:tn
                if cache[i] > maxindex maxindex = cache[i] end
            end
            for i = 1:tn
                cache[i] = maxindex
            end
        end
        index += stride
        sync_threads()
    end
    return nothing
end

#∑∂v∂t, ∑∂ρ∂t, pairs, W, ∇W, ∑W, ∑∇W, ρ, P, v, points, dx, h, h⁻¹, H, H⁻¹, η², m₀, ρ₀, c₀, γ, γ⁻¹,g, δᵩ, α, β, 𝜈, s, dpc_l₀, dpc_pmin, dpc_pmax, dpc_λ, xsph_𝜀, Δt, sphkernel, ptype
#####################################################################
#####################################################################
# SPH
#####################################################################
"""

    W_2d!(W, pairs, sphkernel, H⁻¹) 

Compute kernel values for each particles pair in list. Update `W`. See SPHKernels.jl for details.
"""
function W_2d!(W, pairs, points, H⁻¹, sphkernel) 
    gpukernel = @cuda launch=false kernel_W_2d!(W, pairs, points, H⁻¹, sphkernel) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(W, pairs, points, H⁻¹, sphkernel; threads = Tx, blocks = Bx)
end
function kernel_W_2d!(W, pairs, points, H⁻¹, sphkernel) 
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

    ∑W_2d!(∑W, pairs, sphkernel, H⁻¹) 

Compute sum of kernel values for each particles pair in list. Add to `∑W`. See SPHKernels.jl for details.
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

Compute gradients. Update `∇W`. See SPHKernels.jl for details.

"""
function ∇W_2d!(∇W, pairs, points, H⁻¹, kernel) 
    gpukernel = @cuda launch=false kernel_∇W_2d!(∇W, pairs, points, H⁻¹, kernel) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∇W, pairs, points, H⁻¹, kernel; threads = Tx, blocks = Bx)
end
function kernel_∇W_2d!(∇W, pairs, points, H⁻¹, kernel) 
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

Compute gradients. Add sum to `∑∇W` and update `∇W`. See SPHKernels.jl for details.

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
"""
    
    ∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Add sum to ∑∇W. See SPHKernels.jl for details.

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

#∑∂v∂t, ∑∂ρ∂t, pairs, W, ∇W, ∑W, ∑∇W, ρ, P, v, points, dx, h, h⁻¹, H, H⁻¹, η², m₀, ρ₀, c₀, γ, γ⁻¹, g, δᵩ, α, β, 𝜈, s, dpc_l₀, dpc_pmin, dpc_pmax, dpc_λ, xsph_𝜀, Δt, sphkernel, ptype
#####################################################################
# https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28
@inline function powfancy7th(x, γ⁻¹, γ)
    if γ == 7
        # todo tune the magic constant
        # initial guess based on fast inverse sqrt trick but adjusted to compute x^(1/7)
        t = copysign(reinterpret(Float64, 0x36cd000000000000 + reinterpret(UInt64,abs(x))÷7), x)
        @fastmath for _ in 1:2
        # newton's method for t^3 - x/t^4 = 0
            t2 = t*t
            t3 = t2*t
            t4 = t2*t2
            xot4 = x/t4
            t = t - t*(t3 - xot4)/(4*t3 + 3*xot4)
        end
        return t
    end
    return x^γ⁻¹
end
"""
    
    ∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 

Compute ∂ρ∂t - density derivative includind density diffusion. *Replace all values and update `∑∂ρ∂t`.*

```math

\\frac{\\partial \\rho_i}{\\partial t} = \\sum  m_j \\textbf{v}_{ij} \\cdot \\nabla_i W_{ij} + \\delta_{\\Phi} h c_0 \\sum \\Psi_{ij} \\cdot \\nabla_i W_{ij} \\frac{m_j}{\\rho_j}

\\\\

\\Psi_{ij} = 2 (\\rho_{ij}^T - \\rho_{ij}^H) \\frac{\\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}

\\\\

\\rho_{ij}^H = \\rho_0 \\left( \\sqrt[\\gamma]{\\frac{P_{ij}^H + 1}{C_b}} - 1\\right)

\\\\

P_{ij}^H = \\rho_0 g z_{ij}

```

``z_{ij}`` - vertical distance.

"""
function ∂ρ∂tDDT!(∑∂ρ∂t::CuArray{T}, pairs, ∇W, ρ, v, points, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024)  where T
    fill!(∑∂ρ∂t, zero(T))
    η²    = (0.1*h)*(0.1*h)
    γ⁻¹   = 1/γ
    DDTkh = 2 * h * δᵩ * c₀
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb
    if length(pairs) != length(∇W) error("Length shoul be equal") end

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT!(∑∂ρ∂t, pairs, ∇W, ρ, v, points, η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, pairs, ∇W, ρ, v, points, η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT!(∑∂ρ∂t, pairs, ∇W, ρ, v, points, η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    tindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    index      = tindex
    # move it outside kernel
    #γ⁻¹  = 1/γ
    #η²   = (0.1*h)*(0.1*h)
    
    #DDTkh = 2 * h * δᵩ * c₀

    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ > 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
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
            # if 1 + DDTgz * Δx[2] < 0 || 1 - DDTgz * Δx[2] < 0 return nothing end
            
            m₀dot     = m₀ * (Δv[1] * ∇Wᵢⱼ[1] + Δv[2] * ∇Wᵢⱼ[2])  #  Δv ⋅ ∇Wᵢⱼ
            ∑∂ρ∂ti = ∑∂ρ∂tj = m₀dot

            if ptype[pᵢ] >= 1
                drhopvp = ρ₀ * powfancy7th(1 + DDTgz * Δx[2], γ⁻¹, γ) - ρ₀ 
                #drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀
                visc_densi = DDTkh  * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
                delta_i    = visc_densi * dot3 * m₀ / ρⱼ
                ∑∂ρ∂ti    += delta_i #* (ptype[pᵢ] >= 1)
            end
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti 

            if ptype[pⱼ] >= 1
                drhopvn = ρ₀ * powfancy7th(1 - DDTgz * Δx[2], γ⁻¹, γ) - ρ₀
                #drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
                visc_densi = DDTkh  * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
                delta_j    = visc_densi * dot3 * m₀ / ρᵢ
                ∑∂ρ∂tj    += delta_j #* (ptype[pⱼ] >= 1)
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
    pressure(ρ, c₀, γ, ρ₀)

Equation of State in Weakly-Compressible SPH

```math
P = c_0^2 \\rho_0 * \\left[  \\left( \\frac{\\rho}{\\rho_0} \\right)^{\\gamma}  \\right]
```
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
    #return  P₀ * ((ρ / ρ₀) ^ γ - 1) * (ptype < 1 && ρ < ρ₀)
    if ptype < 0 && ρ < ρ₀
        return 0.0
    end
    return  P₀ * ((ρ / ρ₀) ^ γ - 1)
end

#####################################################################
"""
    
    pressure!(P, ρ, c₀, γ, ρ₀, ptype) 

Equation of State in Weakly-Compressible SPH.

```math
P = c_0^2 \\rho_0 * \\left[  \\left( \\frac{\\rho}{\\rho_0} \\right)^{\\gamma}  \\right]
```
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
    
    ∂v∂t!(∑∂v∂t,  ∇W, pairs, m₀, ρ, c₀, γ, ρ₀) 

The momentum equation (without dissipation and gravity). *Add to `∑∂v∂t`.*

```math
\\frac{\\partial \\textbf{v}_i}{\\partial t} = - \\sum  m_j \\left( \\frac{p_i}{\\rho^2_i} + \\frac{p_j}{\\rho^2_j} \\right) \\nabla_i W_{ij}
```

"""
function ∂v∂t!(∑∂v∂t,  ∇W, P, pairs, m₀, ρ, ptype; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false kernel_∂v∂t!(∑∂v∂t,  ∇W, P, pairs, m₀, ρ, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t,  ∇W, P, pairs, m₀, ρ, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂v∂t!(∑∂v∂t, ∇W, P, pairs, m₀, ρ, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] >= 0 && ptype[pⱼ] >= 0
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Pᵢ    = P[pᵢ]
            Pⱼ    = P[pⱼ]
            ∇Wᵢⱼ  = ∇W[index]
            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            ∂v∂t  = (- m₀ * Pfac * ∇Wᵢⱼ[1], - m₀ * Pfac * ∇Wᵢⱼ[2])
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
    
    ∂v∂t_av!(∑∂v∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀)


Compute artificial viscosity part of ∂v∂t. *Add to `∑∂v∂t`.*

```math
\\Pi_{ij} = \\begin{cases} \\frac{- \\alpha \\overline{c}_{ij} \\mu_{ij} + \\beta \\mu_{ij}^2 }{\\overline{\\rho}_{ij}} &  \\textbf{v}_{ij} \\cdot \\textbf{r}_{ij} < 0 \\\\ 0 &  otherwise \\end{cases}
```

```math
\\mu_{ij} = \\frac{h \\textbf{v}_{ij}\\cdot \\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}
```

```math
\\overline{c}_{ij}  = \\frac{c_i + c_j}{2}
```

```math
\\overline{\\rho}_{ij} = \\frac{\\rho_i + \\rho_j}{2}
```

```math
\\beta = 0

\\c_{ij} = c_0

\\m_i = m_j = m_0

```

Artificial viscosity part of momentum equation. 

```math
\\frac{\\partial \\textbf{v}_i}{\\partial t} = - \\sum  m_j \\Pi_{ij} \\nabla_i W_{ij}
```

J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.

"""
function ∂v∂t_av!(∑∂v∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀, ptype; minthreads::Int = 1024) 
    
    η²    = (0.1 * h) * (0.1 * h)
    gpukernel = @cuda launch=false kernel_∂v∂t_av!(∑∂v∂t, ∇W, pairs, points, h, η², ρ, α, v, c₀, m₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, ∇W, pairs, points, h, η², ρ, α, v, c₀, m₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂v∂t_av!(∑∂v∂t, ∇W, pairs, points, h, η², ρ, α, v, c₀, m₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            #=
            if isnan(ρᵢ) || iszero(ρᵢ) || ρᵢ < 0.001 || isnan(ρⱼ) || iszero(ρⱼ) || ρⱼ < 0.001
                @cuprintln "kernel Π: index =  $index, rhoi = $ρᵢ, rhoi = $ρⱼ, dx = $Δx, r =  $r², pair = $pair"
                error() 
            end
            =#
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            ρₘ     = (ρᵢ + ρⱼ) * 0.5
            ∇Wᵢⱼ   = ∇W[index]
            cond   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            if cond < 0
                Δμ   = h * cond / (r² + η²)
                ΔΠ   =  (-α * c₀ * Δμ) / ρₘ
                ΔΠm₀∇W = (-ΔΠ * m₀ * ∇Wᵢⱼ[1], -ΔΠ * m₀ * ∇Wᵢⱼ[2])
                #=
                if isnan(ΔΠm₀∇W[1])
                    @cuprintln "kernel Π: Π = $ΔΠ ,  W = $(∇W[1])"
                    error() 
                end
                =#
                ∑∂v∂tˣ = ∑∂v∂t[1]
                ∑∂v∂tʸ = ∑∂v∂t[2]   
                CUDA.@atomic ∑∂v∂tˣ[pᵢ] += ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂v∂tʸ[pᵢ] += ΔΠm₀∇W[2]
                CUDA.@atomic ∑∂v∂tˣ[pⱼ] -= ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂v∂tʸ[pⱼ] -= ΔΠm₀∇W[2]
            end
        end
    end
    return nothing
end
#####################################################################
"""
    
    ∂v∂t_visc!(∑∂v∂t,  ∇W, pairs, m, ρ, c₀, γ, ρ₀) 

Compute laminar shear stresse part of ∂v∂t. *Add to `∑∂v∂t`.*

```math
\\frac{\\partial \\textbf{v}_i}{\\partial t} = \\sum \\frac{m_j}{\\rho_j}  \\left( 2 \\nu_i \\frac{\\textbf{r}_{ij} \\cdot \\nabla_i W_{ij} }{r_{ij}^2} \\right) \\textbf{v}_{ij}
```
"""
function ∂v∂t_visc!(∑∂v∂t, ∇W, v, ρ, points, pairs, h, m₀, 𝜈, ptype; minthreads::Int = 1024) 
    η²    = (0.1 * h) * (0.1 * h)
    gpukernel = @cuda launch=false kernel_∂v∂t_visc!(∑∂v∂t, ∇W, v, ρ, points, pairs, η², m₀, 𝜈, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, ∇W, v, ρ, points, pairs, η², m₀, 𝜈, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂v∂t_visc!(∑∂v∂t, ∇W, v, ρ, points, pairs, η², m₀, 𝜈, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            ∇Wᵢⱼ  = ∇W[index]

            𝜈term = 4𝜈 * m₀ * (Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2] ) / ((ρᵢ + ρⱼ) * (r² + η²))  

            ∂v∂t  = (𝜈term * Δv[1], 𝜈term * Δv[2])
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
    
    ∂v∂t_addgrav!(∑∂v∂t, gvec)  

Add gravity to the momentum equation.
"""
function ∂v∂t_addgrav!(∑∂v∂t, gvec) 
    gpukernel = @cuda launch=false kernel_∂v∂t_addgrav!(∑∂v∂t, gvec) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(∑∂v∂t[1])
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, gvec; threads = Tx, blocks = Bx)
end
function kernel_∂v∂t_addgrav!(∑∂v∂t, gvec) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∑∂v∂t[1])
        ∑∂v∂tˣ = ∑∂v∂t[1]
        ∑∂v∂tʸ = ∑∂v∂t[2]
        ∑∂v∂tˣ[index] -= gvec[1]
        ∑∂v∂tʸ[index] -= gvec[2]
        
    end
    return nothing
end
#####################################################################
"""
    update_ρp∂ρ∂tΔt!(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype) 

Update dencity.

```math
\\rho = \\rho + \\frac{\\partial \\rho}{\\partial t} * \\Delta t
```
"""
function update_ρp∂ρ∂tΔt!(ρ, ∑∂ρ∂t, Δt, ρ₀, ptype) 
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
        if ρval < ρ₀ && ptype[index] < 0 ρval = ρ₀ end
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
    update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 

Update vlocity.

```math
\\textbf{v} = \\textbf{v} + \\frac{\\partial \\textbf{v}}{\\partial t} * \\Delta t
```
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

```math
\\textbf{r} = \\textbf{r} +  \\textbf{v} * \\Delta t
```

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
    
    symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 

Symplectic Position Verlet scheme.

* Parshikov et al, 2000
* Leimkuhler and Matthews, 2016

"""
function symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) # << rename
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)
        if ρval < ρ₀ && ptype[index] < 0 ρval = ρ₀ end

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
        if pᵢ != 0 && ptype[pᵢ] >= 0 && ptype[pⱼ] >= 0
           
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

Dynamic Particle Collision (DPC) correction. *Replace all values and update `∑Δvdpc`.*


```math
\\delta \\textbf{v}_i^{DPC} = \\sum k_{ij}\\frac{m_j}{m_i + m_j}v_{ij}^{coll} + \\frac{\\Delta  t}{\\rho_i}\\sum \\phi_{ij} \\frac{2V_j}{V_i + V_j}\\frac{p_{ij}^b}{r_{ij}^2 + \\eta^2}\\textbf{r}_{ij}

\\\\

(v_{ij}^{coll} , \\quad \\phi_{ij}) = \\begin{cases} (\\frac{\\textbf{v}_{ij}\\cdot \\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}\\textbf{r}_{ji}, \\quad 0) & \\textbf{v}_{ij}\\cdot \\textbf{r}_{ij} < 0 \\\\ (0, \\quad 1) &  otherwise \\end{cases}

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
function dpcreg!(∑Δvdpc, v, ρ::CuArray{T}, P::CuArray{T}, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim) where T
    fill!(∑Δvdpc[1], zero(T))
    fill!(∑Δvdpc[2], zero(T))
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
    update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 

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
# Carlos Alberto Dutra Fraga Filho, Reflective Boundary Conditions Coupled With the SPH Method for 
# the Three-Dimensional Simulation of Fluid-Structure Interaction With Solid Boundaries
###################################################################################
"""
    
    cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹)

Corrected Smoothed Particle Method (CSPM) Density Renormalisation.

```math

\\rho_{i}^{norm} = \\frac{\\sum m_j W}{\\sum \\frac{m_j}{\\rho_j} W}
```

Chen JK, Beraun JE, Carney TC (1999) A corrective smoothed particle method for boundary value problems in heat conduction. Int. J. Num. Meth. Engng. https://doi.org/10.1002/(SICI)1097-0207(19990920)46:2<231::AID-NME672>3.0.CO;2-K
"""
function cspmcorr!(∑ρcspm, W, ρ::CuArray{T}, m₀, pairs, ptype) where T
    fill!(∑ρcspm[1], zero(T))
    fill!(∑ρcspm[2], zero(T))

    gpukernel = @cuda launch=false kernel_cspmcorr_1!(∑ρcspm, W, ρ, m₀, pairs, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑ρcspm, W, ρ, m₀, pairs, ptype; threads = Tx, blocks = Bx)

    gpukernel2 = @cuda launch=false kernel_cspmcorr_2!(ρ, ∑ρcspm) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(ρ)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel2(ρ, ∑ρcspm; threads = Tx, blocks = Bx)
end
function kernel_cspmcorr_1!(∑ρcspm, W, ρ, m₀, pairs, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] >= 0 && ptype[pⱼ] >= 0
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            w     = W[index]
            CUDA.@atomic ∑ρcspm[1][pᵢ] +=  m₀ * w
            CUDA.@atomic ∑ρcspm[2][pᵢ] +=  w * m₀ / ρⱼ

            CUDA.@atomic ∑ρcspm[1][pⱼ] +=  m₀ * w
            CUDA.@atomic ∑ρcspm[2][pⱼ] +=  w * m₀ / ρᵢ
        end
    end
    return nothing
end
function kernel_cspmcorr_2!(ρ, ∑ρcspm) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(ρ) 
        newρ = ∑ρcspm[1][index] / ∑ρcspm[2][index]
        if !isnan(newρ) ρ[index] = newρ end
    end
    return nothing
end
#####################################################################
# XSPH Correction 
#####################################################################
"""
    
    xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀)

The XSPH correction.

```math

\\hat{\\textbf{v}_{i}} = - \\epsilon \\sum m_j \\frac{\\textbf{v}_{ij}}{\\overline{\\rho}_{ij}} W_{ij}

```

* Monaghan JJ (1989) On the problem of penetration in particle methods. J Comput Phys. https://doi.org/10.1016/0021-9991(89)90032-6

* Carlos Alberto Dutra Fraga Filho, Reflective Boundary Conditions Coupled With the SPH Method for the Three-Dimensional Simulation of Fluid-Structure Interaction With Solid Boundaries
"""
function xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀)
    fill!(∑Δvxsph[1], zero(T))
    fill!(∑Δvxsph[2], zero(T))
    gpukernel = @cuda launch=false kernel_xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀; threads = Tx, blocks = Bx)
end
function kernel_xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            xsph  = 2m₀ * 𝜀 * W[index] / (ρᵢ + ρⱼ)
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
"""
    update_xsphcorr!(v, ∑Δvxsph, ptype) 

Update velocity.
"""
function update_xsphcorr!(v, ∑Δvxsph, ptype) 
    gpukernel = @cuda launch=false kernel_update_dpcreg!(v, ∑Δvxsph, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, ∑Δvxsph, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_xsphcorr!(v, ∑Δvxsph, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)
        if ptype[index] > 0
            vval = v[index]
            xsph = ∑Δvxsph[index]
            v[index] = (vval[1] + xsph[1], vval[2] + xsph[2])
        end
    end
    return nothing
end
#####################################################################
# * Rapaport D.C., 2004. The art of molecular dynamics simulation.
#
# Carlos Alberto Dutra Fraga Filho Julio Tomás Aquije Chacaltana
# BOUNDARY TREATMENT TECHNIQUES IN SMOOTHED
# PARTICLE HYDRODYNAMICS: IMPLEMENTATIONS IN FLUID
# AND THERMAL SCIENCES AND RESULTS ANALYSIS
#####################################################################
"""
    fbmolforce!(∑∂v∂t, pairs, points, d, r₀, ptype)

The repulsive force exerted by the virtual particle on the fluid particle.


```math
F = D * \\frac{\\left( (\\frac{r_0}{\\textbf{r}_{ij}})^{n_1} - (\\frac{r_0}{\\textbf{r}_{ij}})^{n_2}\\right)}{r_{ij}^2}
```
* Rapaport, 2004

n₁ = 12

n₂ = 4
"""
function fbmolforce!(∑∂v∂t, pairs, points, d, r₀, ptype)
    gpukernel = @cuda launch=false kernel_fbmolforce!(∑∂v∂t, pairs, points, d, r₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, pairs, points, d, r₀, ptype; threads = Tx, blocks = Bx)
end
function kernel_fbmolforce!(∑∂v∂t, pairs, points, d, r₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] * ptype[pⱼ] < 0
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            r     = sqrt(Δx[1]^2 + Δx[2]^2)
            if r < r₀
                Fc    = d * ((r₀ / r)^12 - (r₀ / r)^4) / r² 
                F     = (Δx[1] * Fc, Δx[2] * Fc)
                
                ∑∂v∂tˣ = ∑∂v∂t[1]
                ∑∂v∂tʸ = ∑∂v∂t[2] 

                CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  F[1]
                CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  F[2]
                
                CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  F[1]
                CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  F[2]
            end
        end
    end
    return nothing
end