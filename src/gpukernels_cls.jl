#####################################################################
# GPU KERNELS FOR 2D & 3D
#####################################################################
# CELL LIST
#####################################################################
"""
    cellmap!(pcell, cellpnum, points,  h, offset)  

Map each point to cell and count number of points in each cell.

For each coordinates cell number calculated:

```julia
csᵢ = size(cellpnum, 1) 
p₁  =  (x₁ - offset₁) * h₁⁻¹
pᵢ₁ = ceil(min(max(p₁, 1), csᵢ))
```

"""
function cellmap!(pcell, cellpnum, points,  h, offset)  
    h⁻¹ = @. 1/h
    kernel = @cuda launch=false kernel_cellmap!(pcell, cellpnum, points,  h⁻¹, offset) 
    config = launch_configuration(kernel.fun)
    threads = min(length(first(points)), config.threads)
    blocks = cld(length(first(points)), threads)
    CUDA.@sync kernel(pcell, cellpnum, points,  h⁻¹, offset; threads = threads, blocks = blocks)
end
# 2D case
function kernel_cellmap!(pcell, cellpnum, points::NTuple{2, CuDeviceVector{T, 1}},  h⁻¹, offset) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    csˣ = size(cellpnum, 1) 
    csʸ = size(cellpnum, 2) 
    if i <= length(pcell)
        @fastmath  pˣ =  (points[1][i] - offset[1]) * h⁻¹[1]
        @fastmath  pʸ =  (points[2][i] - offset[2]) * h⁻¹[2]
        iˣ = ceil(Int32, min(max(pˣ, 1), csˣ)) 
        iʸ = ceil(Int32, min(max(pʸ, 1), csʸ))
        # maybe add check:  is particle in simulation range? and include only if in simulation area
        pcell[i] = (iˣ, iʸ)
        CUDA.@atomic cellpnum[iˣ, iʸ] += one(Int32) 
    end
    return nothing
end
# 3D case
function kernel_cellmap!(pcell, cellpnum, points::NTuple{3, CuDeviceVector{T, 1}},  h⁻¹, offset) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    csˣ, csʸ, csᶻ = size(cellpnum) 
    if i <= length(pcell)
        @fastmath  pˣ =  (points[1][i] - offset[1]) * h⁻¹[1]
        @fastmath  pʸ =  (points[2][i] - offset[2]) * h⁻¹[2]
        @fastmath  pᶻ =  (points[3][i] - offset[3]) * h⁻¹[3]
        iˣ = ceil(Int32, min(max(pˣ, 1), csˣ)) 
        iʸ = ceil(Int32, min(max(pʸ, 1), csʸ))
        iᶻ = ceil(Int32, min(max(pᶻ, 1), csᶻ))
        # maybe add check:  is particle in simulation range? and include only if in simulation area
        pcell[i] = (iˣ, iʸ, iᶻ)
        CUDA.@atomic cellpnum[iˣ, iʸ, iᶻ] += one(Int32) 
    end
    return nothing
end
#####################################################################
"""
    fillcells_naive!(celllist, cellpnum, pcell) 
    
Fill cell list with cell. Naive approach. No bound check. Values in `pcell` list shoid be in range of `cellpnum` and `celllist`.
"""
function fillcells_naive!(celllist, cellpnum, pcell) 
    CLs = size(celllist)[2:end]
    if size(cellpnum) != CLs error("cell list dimension $(CLs) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_fillcells_naive!(celllist, cellpnum, pcell) 
    config = launch_configuration(gpukernel.fun)
    threads = min(length(pcell), config.threads)
    blocks = cld(length(pcell), threads)
    CUDA.@sync gpukernel(celllist, cellpnum, pcell; threads = threads, blocks = blocks)
end
# 2D case
function kernel_fillcells_naive!(celllist, cellpnum, pcell::CuDeviceVector{NTuple{2, Int32}, 1}) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pcell)
        iˣ, iʸ = pcell[index]
        n = CUDA.@atomic cellpnum[iˣ, iʸ] += 1
        n += 1
        if n <= size(celllist, 1)
            celllist[n, iˣ, iʸ] = index
        end
    end
    return nothing
end
# 3D case
function kernel_fillcells_naive!(celllist, cellpnum, pcell::CuDeviceVector{NTuple{3, Int32}, 1}) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pcell)
        iˣ, iʸ, iᶻ = pcell[index]
        n = CUDA.@atomic cellpnum[iˣ, iʸ, iᶻ] += 1
        n += 1
        if n <= size(celllist, 1)
            celllist[n, iˣ, iʸ, iᶻ] = index
        end
    end
    return nothing
end
#####################################################################
#####################################################################
"""
    мaxpairs(cellpnum)

Maximum number of pairs.
"""
function мaxpairs(cellpnum)
    if length(size(cellpnum)) == 2
        return мaxpairs_2d(cellpnum)
    elseif length(size(cellpnum)) == 3
        return мaxpairs_3d(cellpnum)
    end
end
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
    indexˣ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexʸ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    Nx, Ny = size(cellpnum)
    if  indexˣ <= Nx && indexʸ <= Ny 
        n = cellpnum[indexˣ, indexʸ] 
        if n > 0
            m         = 0
            neibcellˣ = indexˣ - 1
            neibcellʸ = indexʸ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny 
                m += cellpnum[neibcellˣ, neibcellʸ] 
            end

            neibcellˣ = indexˣ 
            neibcellʸ = indexʸ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny 
                m += cellpnum[neibcellˣ, neibcellʸ] 
            end

            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny 
                m += cellpnum[neibcellˣ, neibcellʸ] 
            end

            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ 
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny 
                m += cellpnum[neibcellˣ, neibcellʸ] 
            end

            val  = Int((n * (n - 1)) * 0.5) + m * n
            CUDA.@atomic cnt[1] += val
        end
    end
    return nothing
end

function мaxpairs_3d(cellpnum)
    cnt        = CUDA.zeros(Int, 1)
    Nx, Ny, Nz = size(cellpnum)
    gpukernel  = @cuda launch=false kernel_мaxpairs_3d!(cellpnum, cnt)
    config     = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx         = min(maxThreads, Nx)
    Ty         = min(fld(maxThreads, Tx), Ny)
    Tz         = min(fld(maxThreads, (Tx * Ty)), Nz)
    Bx, By, Bz = cld(Nx, Tx), cld(Ny, Ty), cld(Nz, Tz) 
    threads    = (Tx, Ty, Tz)
    blocks     = (Bx, By, Bz)
    CUDA.@sync gpukernel(cellpnum, cnt; threads = threads, blocks = blocks)
    CUDA.@allowscalar cnt[1]
end
function kernel_мaxpairs_3d!(cellpnum, cnt)
    indexˣ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexʸ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    indexᶻ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    Nx, Ny, Nz = size(cellpnum)
    if  indexˣ <= Nx && indexʸ <= Ny  && indexᶻ <= Nz 
        n = cellpnum[indexˣ, indexʸ, indexᶻ] 
        if n > 0
            
            m         = 0
            # Z 0
            neibcellˣ = indexˣ - 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ 
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            
            neibcellˣ = indexˣ 
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ 
            neibcellᶻ = indexᶻ
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            
            # Z +1
            neibcellˣ = indexˣ - 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ + 1
            
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            
            neibcellˣ = indexˣ 
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ 
            neibcellᶻ = indexᶻ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ
            neibcellʸ = indexʸ 
            neibcellᶻ = indexᶻ + 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            # Z -1
            neibcellˣ = indexˣ - 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ - 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ 
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ - 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ + 1
            neibcellᶻ = indexᶻ - 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end
            neibcellˣ = indexˣ + 1
            neibcellʸ = indexʸ 
            neibcellᶻ = indexᶻ - 1
            if  0 < neibcellˣ <= Nx && 0 < neibcellʸ <= Ny &&  0 < neibcellᶻ <= Nz
                m += cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ] 
            end

            val  = Int((n * (n - 1)) * 0.5) + m * n
            CUDA.@atomic cnt[1] += val
            
        end
    end
    return nothing
end
#####################################################################
"""
    neib_internal!(pairs, cnt, cellpnum, points, celllist, dist)

Find all pairs with distance < h in one cell.
"""
function neib_internal!(pairs, cnt, cellpnum, points::NTuple{2, CuArray{T}}, celllist, dist) where T
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension ($((CLx, CLy))) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_internal!(pairs, cnt, cellpnum, points, celllist, dist², 6)
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
function kernel_neib_internal!(pairs, cnt, cellpnum, points::NTuple{2, CuDeviceVector{T, 1}}, celllist, dist², cs)  where T # cs - cache length for each thread
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
            pᵢ  = (points[1][indi], points[2][indi])
            for j = i + 1:len
                indj = celllist[j, indexᵢ, indexⱼ]
                pⱼ   = (points[1][indj], points[2][indj])
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
function neib_internal!(pairs, cnt, cellpnum, points::NTuple{3, CuArray{T}}, celllist, dist) where T
    dist² = dist^2
    CLn, CLx, CLy, CLz = size(celllist)
    Nx, Ny, Nz = size(cellpnum)
    if (Nx, Ny, Nz) != (CLx, CLy, CLz) error("cell list dimension ($((CLx, CLy, CLz))) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_internal!(pairs, cnt, cellpnum, points, celllist, dist², 6)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx         = min(maxThreads, Nx)
    Ty         = min(fld(maxThreads, Tx), Ny)
    Tz         = min(fld(maxThreads, (Tx * Ty)), Nz)
    Bx, By, Bz = cld(Nx, Tx), cld(Ny, Ty), cld(Nz, Tz) 
    threads    = (Tx, Ty, Tz)
    blocks     = (Bx, By, Bz)
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * Ty * Tz * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, dist², cs; threads = threads, blocks = blocks, shmem= Tx * Ty * Tz * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_internal!(pairs, cnt, cellpnum, points::NTuple{3, CuDeviceVector{T, 1}}, celllist, dist², cs)  where T
    threadx = threadIdx().x
    thready = threadIdx().y
    threadz = threadIdx().z
    indexˣ = (blockIdx().x - Int32(1)) * blockDim().x + threadx
    indexʸ = (blockIdx().y - Int32(1)) * blockDim().y + thready
    indexᶻ = (blockIdx().z - Int32(1)) * blockDim().z + threadz
    Nx, Ny, Nz = size(cellpnum)
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x , blockDim().y, blockDim().z))
    if indexˣ <= Nx && indexʸ <= Ny && indexᶻ <= Nz && cellpnum[indexˣ, indexʸ, indexᶻ] > 1 
        ccnt = zero(Int32)
        len  = cellpnum[indexˣ, indexʸ, indexᶻ]
        for i = 1:len - 1
            indᵢ  = celllist[i, indexˣ, indexʸ, indexᶻ]
            pᵢ  = (points[1][indᵢ], points[2][indᵢ], points[3][indᵢ])
            for j = i + 1:len
                indⱼ = celllist[j, indexˣ, indexʸ, indexᶻ]
                pⱼ   = (points[1][indⱼ], points[2][indⱼ], points[3][indⱼ])
                distance² = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2 + (pᵢ[3] - pⱼ[3])^2
                if distance² < dist²
                    ccnt += 1
                    cache[ccnt, threadx, thready, threadz] = minmax(indᵢ, indⱼ)
                    if ccnt == cs
                        s  = CUDA.@atomic cnt[1] += ccnt
                        if s + ccnt <=length(pairs)
                            for cind in 1:ccnt
                                pairs[s + cind] = cache[cind, threadx, thready, threadz]
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
                    pairs[s + cind] = cache[cind, threadx, thready, threadz]
                end
            end
        end
    end
    return nothing
end
#####################################################################
"""
    neib_external!(pairs, cnt, cellpnum, points, celllist, offset, dist)

Find all pairs with another cell shifted on offset.
"""
function neib_external!(pairs, cnt, cellpnum, points::NTuple{2, CuArray{T}}, celllist, offset, dist) where T
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external!(pairs, cnt, cellpnum, points, celllist,  offset, dist², 6)
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
function kernel_neib_external!(pairs, cnt, cellpnum, points::NTuple{2, CuDeviceVector{T, 1}}, celllist,  offset, dist², cs) where T
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
            pᵢ = (points[1][i], points[2][i])
            for j in jinds
                pⱼ = (points[1][j], points[2][j])
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
# 3D
function neib_external!(pairs, cnt, cellpnum, points::NTuple{3, CuArray{T}}, celllist, offset, dist) where T
    dist² = dist^2
    CLn, CLx, CLy, CLz = size(celllist)
    Nx, Ny, Nz = size(cellpnum)
    if (Nx, Ny, Nz) != (CLx, CLy, CLz) error("cell list dimension $((CLx, CLy, CLz)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external!(pairs, cnt, cellpnum, points, celllist,  offset, dist², 6)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx         = min(maxThreads, Nx)
    Ty         = min(fld(maxThreads, Tx), Ny)
    Tz         = min(fld(maxThreads, (Tx * Ty)), Nz)
    Bx, By, Bz = cld(Nx, Tx), cld(Ny, Ty), cld(Nz, Tz) 
    threads    = (Tx, Ty, Tz)
    blocks     = (Bx, By, Bz)
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * Ty * Tz * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, offset, dist², cs; threads = threads, blocks = blocks, shmem= Tx * Ty * Tz * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_external!(pairs, cnt, cellpnum, points::NTuple{3, CuDeviceVector{T, 1}}, celllist,  offset, dist², cs) where T
    threadx = threadIdx().x
    thready = threadIdx().y
    threadz = threadIdx().z
    indexˣ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    indexʸ = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y 
    indexᶻ = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x, blockDim().y, blockDim().z))
    Nx, Ny, Nz = size(cellpnum)
    neibcellˣ = indexˣ + offset[1]
    neibcellʸ = indexʸ + offset[2]
    neibcellᶻ = indexᶻ + offset[3]
    if 0 < neibcellˣ <= Nx &&  0 < neibcellʸ <= Ny && 0 < neibcellᶻ <= Nz && indexˣ <= Nx && indexʸ <= Ny && indexᶻ <= Nz 
        ccnt  = zero(Int32)
        cpn   = cellpnum[indexˣ, indexʸ, indexᶻ]
        if cpn > 0
            indsᵢ = view(celllist, 1:cpn, indexˣ, indexʸ, indexᶻ)
            indsⱼ = view(celllist, 1:cellpnum[neibcellˣ, neibcellʸ, neibcellᶻ], neibcellˣ, neibcellʸ, neibcellᶻ)
            for i in indsᵢ
                pᵢ = (points[1][i], points[2][i], points[3][i])
                for j in indsⱼ
                    pⱼ = (points[1][j], points[2][j], points[3][j])
                    distance² = (pᵢ[1] - pⱼ[1])^2 + (pᵢ[2] - pⱼ[2])^2 + (pᵢ[3] - pⱼ[3])^2
                    if distance² < dist²
                        ccnt += 1
                        cache[ccnt, threadx, thready, threadz] = minmax(i, j)
                        if ccnt == cs
                            s  = CUDA.@atomic cnt[1] += ccnt
                            if s + ccnt <=length(pairs)
                                for cind in 1:ccnt
                                pairs[s + cind] = cache[cind, threadx, thready, threadz]
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
                        pairs[s + cind] = cache[cind, threadx, thready, threadz]
                    end
                end
            end
        end
    end
    return nothing
end
#####################################################################
"""
    neib_search!(pairs, cnt, cellpnum, points, celllist, dist) 

Search all pairs.
"""
function neib_search!(pairs, cnt, cellpnum, points::NTuple{2, CuArray{T}}, celllist, dist) where T
    fill!(cnt, zero(Int32))                                                            # fill cell pairs counter before neib calk
    neib_internal!(pairs, cnt, cellpnum, points, celllist, dist)                    # modify cnt, pairs < add pairs inside cell
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1, -1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (-1, 1) in grid
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (0,  1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (0, 1) in grid
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  1), dist)          # modify cnt, pairs < add pairs between cell and neiborhood cell by shift (1, 1) in grid
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  0), dist) 
end
function neib_search!(pairs, cnt, cellpnum, points::NTuple{3, CuArray{T}}, celllist, dist) where T
    fill!(cnt, zero(Int32))                                                          
    neib_internal!(pairs, cnt, cellpnum, points, celllist, dist)                   
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1, -1, 0), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (0,  1, 0), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  1, 0), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  0, 0), dist)

    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1, -1, 1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (0,  1, 1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  1, 1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  0, 1), dist) 
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (0,  0, 1), dist)

    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1, -1, -1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (0,  1, -1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  1, -1), dist)          
    neib_external!(pairs, cnt, cellpnum, points, celllist,  (1,  0, -1), dist)
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

