#=
function kernel_cellmap_2d!(pcell, points,  h⁻¹, offset) 
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if i <= length(points)
        @fastmath  p₁ =  (points[i][1] - offset[1]) * h⁻¹[1]
        @fastmath  p₂ =  (points[i][2] - offset[2]) * h⁻¹[2]
        
        pᵢ₁ = ceil(Int32, min(max(p₁, 0.5), 1000.0))
       
        pᵢ₂ = ceil(Int32, min(max(p₂, 0.5), 1000.0))

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
        pᵢ₁ = ceil(Int32, min(max(p₁, 0.5), csᵢ)) 
        pᵢ₂ = ceil(Int32, min(max(p₂, 0.5), csⱼ))
        #if pᵢ₁ <= 0  pᵢ₁  = 1  end
        #if pᵢ₁ > csᵢ pᵢ₁ = csᵢ end

        #if pᵢ₂ <= 0  pᵢ₂  = 1   end
        #if pᵢ₂ > csⱼ pᵢ₂  = csⱼ end

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
=#

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
#=
function ∇Wfunc(αD, q, h) 
    if 0 < q < 2
        return αD * 5 * (q - 2) ^ 3 * q / (8h * (q * h + 1e-6)) 
    end
    return 0.0
end
=#
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