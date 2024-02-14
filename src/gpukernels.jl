#####################################################################
# CELL LIST
#####################################################################
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

#####################################################################


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

#####################################################################
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
        len = cellpnum[indexᵢ, indexⱼ]
        for i = 1:len - 1
            indi = celllist[i, indexᵢ, indexⱼ]
            for j = i + 1:len
                indj = celllist[j, indexᵢ, indexⱼ]
                distance = sqrt((points[indi][1] - points[indj][1])^2 + (points[indi][2] - points[indj][2])^2)
                if distance < dist
                    n = CUDA.@atomic cnt[1] += 1
                    n += 1 
                    if n <= legth(pairs)
                        pairs[n] = tuple(indi, indj, distance)
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
                distance = sqrt((points[i][1] - points[j][1])^2 + (points[i][2] - points[j][2])^2)
                if distance < dist
                    n = CUDA.@atomic cnt[1] += 1
                    n +=1
                    if n <= length(pairs)
                        pairs[n] = tuple(i, j, distance)
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
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end

            # move it outside kernel
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
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇W   = ∇Wₙ[index]
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            dot3  = -(Δx[1] * ∇W[1] + Δx[2] * ∇W[2]) #  - Δx ⋅ ∇W

            # as actual range at timestep Δt½  may be greateg  - some problems can be here
            if 1 + DDTgz * Δx[2] < 0 || 1 - DDTgz * Δx[2] < 0 return nothing end
            
            drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀ ## << CHECK
            visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
            delta_i    = visc_densi * dot3 * m₀ / ρⱼ

            drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
            visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            delta_j    = visc_densi * dot3 * m₀ / ρᵢ

            m₀dot     = m₀ * (Δv[1] * ∇W[1] + Δv[2] * ∇W[2])  #  Δv ⋅ ∇W
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
            ∑∂ρ∂ti = m₀dot + delta_i *  MotionLimiter[pᵢ]
            ∑∂ρ∂tj = m₀dot + delta_j *  MotionLimiter[pⱼ]
            ∑∂ρ∂tval1 = CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti
            ∑∂ρ∂tval2 = CUDA.@atomic ∑∂ρ∂t[pⱼ] += ∑∂ρ∂tj
            #=
            if isnan(ρᵢ) || iszero(ρᵢ) || ρᵢ < 0.001 || isnan(ρⱼ) || iszero(ρⱼ) || ρⱼ < 0.001
                @cuprintln "kernel DDT rho index =  $index , rhoi = $ρᵢ , rhoi = $ρⱼ, dx = $Δx , r =  $r², val1 = $∑∂ρ∂tval1 ,   val2 = $∑∂ρ∂tval2 , pair = $pair"
                error() 
            end

            if isnan(∑∂ρ∂tval1) || isnan(∑∂ρ∂tval2)
                @cuprintln "kernel DDT 3 val1 = $(∑∂ρ∂tval1), val2 = $(∑∂ρ∂tval2), dx1 = $(Δx[1]) , dx2 = $(Δx[2]) rhoi = $ρᵢ , dot3 = $dot3 , visc_densi = $visc_densi drhopvn = $drhopvn $(∇W[1]) $(Δv[1])"
                error() 
            end
            =#
        end
    end
    return nothing
end
"""
    
    ∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 

Compute ∂ρ∂t - density derivative includind density diffusion.
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
                @cuprintln "kernel Π index =  $index , rhoi = $ρᵢ , rhoi = $ρⱼ, dx = $Δx , r =  $r², pair = $pair"
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
                #=
                if isnan(ΔΠm₀∇W[1])
                    @cuprintln "kernel Π: Π = $ΔΠ ,  W = $(∇W[1])"
                    error() 
                end
                =#
                CUDA.@atomic ∑∂Π∂t[pᵢ, 1] += ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂Π∂t[pᵢ, 2] += ΔΠm₀∇W[2]
                CUDA.@atomic ∑∂Π∂t[pⱼ, 1] -= ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂Π∂t[pⱼ, 2] -= ΔΠm₀∇W[2]
            end
        end
    end
    return nothing
end
"""
    
    ∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀)


Compute ∂Π∂t - artificial viscosity.
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
#####################################################################


"""
    pressure(ρ, c₀, γ, ρ₀)

Equation of State in Weakly-Compressible SPH
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
            #=
            if isnan(ρᵢ) || iszero(ρᵢ) || ρᵢ < 0.001 || isnan(ρⱼ) || iszero(ρⱼ) || ρⱼ < 0.001
                @cuprintln "kernel update rho: index =  $index , rhoi = $ρᵢ , rhoi = $ρⱼ, dpdt =  $(∑∂v∂t[index]), pair = $pair"
                error() 
            end
            =#
            Pᵢ    = pressure(ρᵢ, c₀, γ, ρ₀)
            Pⱼ    = pressure(ρⱼ, c₀, γ, ρ₀)
            ∇W    = ∇Wₙ[index]

            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)

            ∂v∂t  = (- m * Pfac * ∇W[1], - m * Pfac * ∇W[2])
            
            if isnan(∂v∂t[1])
                @cuprintln "kernel dvdt: rhoi = $ρᵢ , Pi =  $Pᵢ , m = $m , Pfac = $Pfac , W1 = $(∇W[1])"
                error() 
            end
            
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

The momentum equation (without dissipation).
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

#####################################################################

function kernel_completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= size(∑∂v∂t, 1)
        ∑∂v∂t[index, 1] +=  ∑∂Π∂t[index, 1] - gvec[1] #* gfac[index]
        ∑∂v∂t[index, 2] +=  ∑∂Π∂t[index, 2] - gvec[2] #* gfac[index]
    end
    return nothing
end
"""
    
    completed_∂vᵢ∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac)  

Add gravity and artificial viscosity to the momentum equation.
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
        #=
        if isnan(ρval) || iszero(ρval) || ρval < 0.001
            @cuprintln "kernel update rho: index =  $index , rhoval = $ρval  ,rhoi = $(ρ[index]) , dpdt =  $(∑∂ρ∂t[index]), dt = $Δt , isboundary = $(isboundary[index])"
            error() 
        end
        =#
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
        val = v[index]
        v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        #=
        if isnan(v[index][1] )
            @cuprintln "kernel update v by dvdvt: val = $(val[1]) , dvdt =  $(∑∂v∂t[index, 1] ), dt =  $Δt"
            error() 
        end
        =#
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

function kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, isboundary, ml) # << rename
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(x)

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)
        if ρval < ρ₀ && isboundary[index] ρval = ρ₀ end

        #=
        if isnan(ρval) || iszero(ρval) || ρval < 0.01
            @cuprintln "kernel update all rho: rhova = $ρval , epsi = $epsi , drhodt = $(∑∂ρ∂t[index]) , rhot12 = $(ρΔt½[index]) $Δt"
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
        vval = v[index]
        nval = (vval[1] + ∑∂v∂t[index, 1] * Δt * ml[index], vval[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        vΔt½[index] = nval
        v[index] = nval

        xval = x[index]
        Δxˣ, Δxʸ  = (vval[1] + nval[1]) * 0.5  * Δt, (vval[2] + nval[2]) * 0.5  * Δt
        cΔx[1][index]  += Δxˣ
        cΔx[2][index]  += Δxʸ
        xval = (xval[1] + Δxˣ, xval[2] + Δxʸ)
        xΔt½[index] = xval
        x[index] = xval
    end
    return nothing
end
"""
    
    update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 


"""
function update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, isboundary, ml) 
    if length(x) != length(v) error("Wrong length") end
    gpukernel = @cuda launch=false kernel_update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, isboundary, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(x)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, isboundary, ml; threads = Tx, blocks = Bx)
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

#####################################################################
#####################################################################
function kernel_∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, isboundary) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)
            if !isboundary[pᵢ] && !isboundary[pⱼ]
                xᵢ    = points[pᵢ]
                xⱼ    = points[pⱼ]
                Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
                r     = sqrt(Δx[1]^2 + Δx[2]^2) 
                if r < 2h
                    scos = s * cos(1.5π * r / 2h)/ (r + (0.1*h))
                    CUDA.@atomic ∑∂v∂t[pᵢ, 1] +=  scos * Δx[1] / m₀
                    CUDA.@atomic ∑∂v∂t[pᵢ, 2] +=  scos * Δx[2] / m₀
                    CUDA.@atomic ∑∂v∂t[pⱼ, 1] -=  scos * Δx[1] / m₀
                    CUDA.@atomic ∑∂v∂t[pⱼ, 2] -=  scos * Δx[2] / m₀
                end
            end
        end
    end
    return nothing
end
"""
    
    ∂v∂tpF!(∑∂v∂t, pairs, points, s, H) 

Add surface tension to ∑∂v∂t. Modified.

A. Tartakovsky and P. Meakin, Phys. Rev. E 72 (2005)
"""
function ∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, isboundary) 
    gpukernel = @cuda launch=false kernel_∂v∂tpF!(∑∂v∂t, pairs, points, s, h, m₀, isboundary) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂v∂t, pairs, points, s, h, m₀, isboundary; threads = Tx, blocks = Bx)
end