
#####################################################################
# Make neighbor matrix (list) EXPERIMENTAL
#####################################################################
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
function kernel_∂ρ∂tDDT_2!(∑∂ρ∂t, nlist, ncnt, points, kernel, h, H⁻¹, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, isboundary) 
    
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(points)
        pᵢ    = index
        xᵢ    = points[pᵢ]
        ρᵢ    = ρ[pᵢ]
        vᵢ    = v[pᵢ]

        γ⁻¹   = 1/γ
        η²    = (0.1*h)*(0.1*h)
        Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
        DDTgz = ρ₀ * g / Cb
        DDTkh = 2 * h * δᵩ
        #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
            =#
        for n = 1:ncnt[index]

            pⱼ    = nlist[n]
            xⱼ    = points[pⱼ]
            ρⱼ    = ρ[pⱼ]
            vⱼ    = v[pⱼ]

            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 

            Δv    = (vᵢ[1] - vⱼ[1], vᵢ[2] - vⱼ[2])

            # caclulate ∇W
            r     = sqrt(r²) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇W    = (Δx[1] * dwk_r, Δx[2] * dwk_r)

            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
        
            ∂ρ∂ti     = m₀ * (Δv[1] * ∇W[1] + Δv[2] * ∇W[2])  #  Δv ⋅ ∇W
            
            DDTgxΔx = 1 + DDTgz * Δx[2] 
            # as actual range at timestep Δt½  may be greateg  - some problems can be here
            if !isboundary[pᵢ] && DDTgxΔx >= 0
                dot3       = -(Δx[1] * ∇W[1] + Δx[2] * ∇W[2]) #  - Δx ⋅ ∇W
                drhopvp    = ρ₀ * (DDTgxΔx)^γ⁻¹ - ρ₀ ## << CHECK
                visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
                delta_i    = visc_densi * dot3 * m₀ / ρⱼ
                ∂ρ∂ti     += delta_i 
            end
            ∑∂ρ∂t[pᵢ] += ∂ρ∂ti 
        end
    end
    
    return nothing
end
function ∂ρ∂tDDT_2!(∑∂ρ∂t, nlist, ncnt, points, kernel, h, H⁻¹, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, isboundary) 
    if size(nlist, 2) != length(points) error("Length shoul be equal") end

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT_2!(∑∂ρ∂t, nlist, ncnt, points, kernel, h, H⁻¹, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, isboundary) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(points)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, nlist, ncnt, points, kernel, h, H⁻¹, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, isboundary; threads = Tx, blocks = Bx)
end