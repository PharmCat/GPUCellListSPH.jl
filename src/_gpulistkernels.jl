
#####################################################################
# Make neighbor matrix (list) EXPERIMENTAL
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
            for i = 1:celln
                indexj = clist[i]
                pointj = points[indexj]
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


        #=
function neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², 6)
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx) 
    Bx  = 1 # Blocks in grid.
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, offset, dist², cs; threads = Tx, blocks = Bx, shmem = Tx * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², cs)
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Nx, Ny = size(cellpnum)
    scnt   = CuStaticSharedArray(Int32, 1)
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x))

    if threadIdx().x == 1
        scnt[1] = cnt[1]
    end
    sync_threads()
    Nx, Ny = size(cellpnum)
    while index <= length(cellpnum)
        indexⱼ    = cld(index, Nx)             # y
        indexᵢ    = index - Nx * (indexⱼ - 1)  # x
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
                        ccnt += 1
                        cache[ccnt, threadIdx().x] = minmax(i, j)
                        if ccnt == cs
                            s  = CUDA.@atomic scnt[1] += ccnt
                            if s + ccnt <=length(pairs)
                                for cind in 1:ccnt
                                    pairs[s + cind] = cache[cind, threadIdx().x]
                                end
                            end
                            ccnt = 0
                        end 
                    end
                end  
            end        
            if ccnt > 0 
                s  = CUDA.@atomic scnt[1] += ccnt
                if s + ccnt <=length(pairs)
                    for cind in 1:ccnt
                        pairs[s + cind] = cache[cind, threadIdx().x]
                    end
                end
            end
        end
    index += stride
    end
    sync_threads()
    if threadIdx().x == 1 
        cnt[1] = scnt[1]
    end
    return nothing
end
=#

#=
function neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)
    dist² = dist^2
    CLn, CLx, CLy = size(celllist)
    Nx, Ny = size(cellpnum)
    if (Nx, Ny) != (CLx, CLy) error("cell list dimension $((CLx, CLy)) not equal cellpnum $(size(cellpnum))...") end
    gpukernel = @cuda launch=false kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², 6)  max_registers=32
    config = launch_configuration(gpukernel.fun)
    maxThreads = config.threads
    Tx  = min(32, maxThreads, Nx) 
    Bx  = 8# cld(Nx, Tx) # Blocks in grid.
    cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),  Tx * sizeof(Tuple{Int32, Int32}))
    CUDA.@sync gpukernel(pairs, cnt, cellpnum, points, celllist, offset, dist², cs; threads = Tx, blocks = Bx, shmem = Tx * cs * sizeof(Tuple{Int32, Int32}))
end
function kernel_neib_external_2d!(pairs, cnt, cellpnum, points, celllist,  offset, dist², cs)
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Nx, Ny = size(cellpnum)
    cache  = CuDynamicSharedArray(Tuple{Int32, Int32}, (cs, blockDim().x))

    Nx, Ny = size(cellpnum)
    while index <= length(cellpnum)
        indexⱼ    = cld(index, Nx)             # y
        indexᵢ    = index - Nx * (indexⱼ - 1)  # x
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
                        ccnt += 1
                        cache[ccnt, threadIdx().x] = minmax(i, j)
                        if ccnt == cs
                            s  = CUDA.@atomic cnt[1] += ccnt
                            if s + ccnt <=length(pairs)
                                for cind in 1:ccnt
                                    pairs[s + cind] = cache[cind, threadIdx().x]
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
                        pairs[s + cind] = cache[cind, threadIdx().x]
                    end
                end
            end
        end
    index += stride
    end
    return nothing
end
=#