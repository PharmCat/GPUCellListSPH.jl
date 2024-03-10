
function getsvec(vecs::NTuple{2, <: AbstractVector}, index)
    SVector((vecs[1][index], vecs[2][index]))
end
function getsvec(vecs::NTuple{3, <: AbstractVector}, index) 
    SVector((vecs[1][index], vecs[2][index], vecs[3][index]))
end
function setsvec!(vecs::NTuple{2, <: AbstractVector}, vals, index)
    vecs[1][index] = vals[1]
    vecs[2][index] = vals[2]
end
function setsvec!(vecs::NTuple{3, <: AbstractVector}, vals, index) 
    vecs[1][index] = vals[1]
    vecs[2][index] = vals[2]
    vecs[3][index] = vals[3]
end
function addsvec!(vecs::NTuple{2, <: AbstractVector}, vals, index)
    vecs[1][index] += vals[1]
    vecs[2][index] += vals[2]
end

function addsvec!(vecs::NTuple{3, <: AbstractVector}, vals, index) 
    vecs[1][index] += vals[1]
    vecs[2][index] += vals[2]
    vecs[3][index] += vals[3]
end
function subsvec!(vecs::NTuple{2, <: AbstractVector}, vals, index)
    vecs[1][index] -= vals[1]
    vecs[2][index] -= vals[2]
end
function subsvec!(vecs::NTuple{3, <: AbstractVector}, vals, index) 
    vecs[1][index] -= vals[1]
    vecs[2][index] -= vals[2]
    vecs[3][index] -= vals[3]
end
function atomicaddsvec!(vecs::NTuple{2, <: AbstractVector}, arg, index)
    vec1 = vecs[1]
    vec2 = vecs[2]
    CUDA.@atomic vec1[index] += arg[1]
    CUDA.@atomic vec2[index] += arg[2]
end
function atomicaddsvec!(vecs::NTuple{3, <: AbstractVector}, arg, index)
    vec1 = vecs[1]
    vec2 = vecs[2]
    vec3 = vecs[3]
    CUDA.@atomic vec1[index] += arg[1]
    CUDA.@atomic vec2[index] += arg[2]
    CUDA.@atomic vec3[index] += arg[3]
end
function atomicsubsvec!(vecs::NTuple{2, <: AbstractVector}, arg, index)
    vec1 = vecs[1]
    vec2 = vecs[2]
    CUDA.@atomic vec1[index] -= arg[1]
    CUDA.@atomic vec2[index] -= arg[2]
end
function atomicsubsvec!(vecs::NTuple{3, <: AbstractVector}, arg, index)
    vec1 = vecs[1]
    vec2 = vecs[2]
    vec3 = vecs[3]
    CUDA.@atomic vec1[index] -= arg[1]
    CUDA.@atomic vec2[index] -= arg[2]
    CUDA.@atomic vec3[index] -= arg[3]
end
#∑∂v∂t, ∑∂ρ∂t, pairs, W, ∇W, ∑W, ∑∇W, ρ, P, v, points, dx, h, h⁻¹, H, H⁻¹, η², m₀, ρ₀, c₀, γ, γ⁻¹,g, δᵩ, α, β, 𝜈, s, dpc_l₀, dpc_pmin, dpc_pmax, dpc_λ, xsph_𝜀, Δt, sphkernel, ptype
#####################################################################
#####################################################################
# SPH
#####################################################################
"""

    sphW!(W, pairs, sphkernel, H⁻¹) 

Compute kernel values for each particles pair in list. Update `W`. See SPHKernels.jl for details.
"""
function sphW!(W, pairs, points, H⁻¹, sphkernel) 
    gpukernel = @cuda launch=false kernel_sphW!(W, pairs, points, H⁻¹, sphkernel) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(W, pairs, points, H⁻¹, sphkernel; threads = Tx, blocks = Bx)
end
function kernel_sphW!(W, pairs, points, H⁻¹, sphkernel) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u        = r * H⁻¹
            w        = 𝒲(sphkernel, u, H⁻¹)
            W[index] = w
        end
    end
    return nothing
end
#=
function kernel_sphW!(W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, H⁻¹, sphkernel) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ       = (points[1][pᵢ], points[2][pᵢ], points[3][pᵢ])
            xⱼ       = (points[1][pⱼ], points[2][pⱼ], points[3][pⱼ])
            Δx       = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2], xᵢ[3] - xⱼ[3])
            r        = sqrt(Δx[1]^2 + Δx[2]^2 + Δx[3]^2) 
            u        = r * H⁻¹
            w        = 𝒲(sphkernel, u, H⁻¹)
            W[index] = w
        end
    end
    return nothing
end
=#
#####################################################################
#
#####################################################################
"""

    sph∑W!(∑W, pairs, sphkernel, H⁻¹) 

Compute sum of kernel values for each particles pair in list. Add to `∑W`. See SPHKernels.jl for details.
"""
function sph∑W!(∑W, pairs, points, sphkernel, H⁻¹, ptype) 
    gpukernel = @cuda launch=false kernel_sph∑W!(∑W, pairs, points, sphkernel, H⁻¹, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑W, pairs, points, sphkernel, H⁻¹, ptype; threads = Tx, blocks = Bx)
end
# 2D
function kernel_sph∑W!(∑W, pairs, points, sphkernel, H⁻¹, ptype)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && !((ptype[pᵢ] > 0) ⊻ (ptype[pⱼ] > 0))
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u     = r * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic ∑W[pᵢ] += w
            CUDA.@atomic ∑W[pⱼ] += w
        end
    end
    return nothing
end
# 3D
#=
function kernel_sph∑W!(∑W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, sphkernel, H⁻¹, ptype) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] >= 0 && ptype[pⱼ] >= 0 
            xᵢ       = (points[1][pᵢ], points[2][pᵢ], points[3][pᵢ])
            xⱼ       = (points[1][pⱼ], points[2][pⱼ], points[3][pⱼ])
            Δx       = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2], xᵢ[3] - xⱼ[3])
            r        = sqrt(Δx[1]^2 + Δx[2]^2 + Δx[3]^2) 
            u     = r * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic ∑W[pᵢ] += w
            CUDA.@atomic ∑W[pⱼ] += w
        end
    end
    return nothing
end
=#
#####################################################################
"""
    
    sph∇W!(∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Update `∇W`. See SPHKernels.jl for details.

"""
function sph∇W!(∇W, pairs, points, H⁻¹, kernel) 
    gpukernel = @cuda launch=false kernel_sph∇W!(∇W, pairs, points, H⁻¹, kernel) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∇W, pairs, points, H⁻¹, kernel; threads = Tx, blocks = Bx)
end
# 2D
function kernel_sph∇W!(∇W, pairs, points, H⁻¹, kernel)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            dwr   = Δx * dwk_r
            setsvec!(∇W, dwr, index)
        end
    end
    return nothing
end
# 3D
#=
function kernel_sph∇W!(∇W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, H⁻¹, kernel) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u         = r * H⁻¹
            dwk_r     = d𝒲(kernel, u, H⁻¹) / r
            ∇W[1][index] = Δx[1] * dwk_r
            ∇W[2][index] = Δx[2] * dwk_r
            ∇W[3][index] = Δx[3] * dwk_r
        end
    end
    return nothing
end
=#
#####################################################################
#=
"""
    
    sph∑∇W!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Add sum to `∑∇W` and update `∇W`. See SPHKernels.jl for details.

"""
function sph∑∇W!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_sph∑∇W!(∑∇W, ∇W, pairs, points, kernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∇W, ∇W, pairs, points, kernel, H⁻¹; threads = Tx, blocks = Bx)
end
# 2D
function kernel_sph∑∇W!(∑∇W, ∇W, pairs, points::NTuple{2, CuDeviceVector{T, 1}}, kernel, H⁻¹) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ    = (points[1][pᵢ], points[2][pᵢ])
            xⱼ    = (points[1][pⱼ], points[2][pⱼ])
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
# 3D
function kernel_sph∑∇W!(∑∇W, ∇W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, kernel, H⁻¹) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            xᵢ       = (points[1][pᵢ], points[2][pᵢ], points[3][pᵢ])
            xⱼ       = (points[1][pⱼ], points[2][pⱼ], points[3][pⱼ])
            Δx       = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2], xᵢ[3] - xⱼ[3])
            r        = sqrt(Δx[1]^2 + Δx[2]^2 + Δx[3]^2) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇w    = (Δx[1] * dwk_r, Δx[2] * dwk_r, Δx[3] * dwk_r)
            ∑∇Wˣ = ∑∇W[1]
            ∑∇Wʸ = ∑∇W[2]
            ∑∇Wᶻ = ∑∇W[3]
            CUDA.@atomic ∑∇Wˣ[pᵢ] += ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pᵢ] += ∇w[2]
            CUDA.@atomic ∑∇Wᶻ[pᵢ] += ∇w[3]

            CUDA.@atomic ∑∇Wˣ[pⱼ] -= ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pⱼ] -= ∇w[2]
            CUDA.@atomic ∑∇Wᶻ[pⱼ] -= ∇w[3]

            ∇W[index] = ∇w
        end
    end
    return nothing
end
=#
#####################################################################
"""
    
    sph∑∇W!(∑∇W, pairs, points, kernel, H⁻¹) 

Compute gradients. Add sum to ∑∇W. See SPHKernels.jl for details.

"""
function sph∑∇W!(∑∇W, pairs, points, kernel, H⁻¹, ptype) 
    gpukernel = @cuda launch=false kernel_sph∑∇W!(∑∇W, pairs, points, kernel, H⁻¹, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∇W, pairs, points, kernel, H⁻¹, ptype; threads = Tx, blocks = Bx)
end
# 2D
function kernel_sph∑∇W!(∑∇W, pairs, points, kernel, H⁻¹, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && !((ptype[pᵢ] > 0) ⊻ (ptype[pⱼ] > 0))
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u     = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇w    = Δx * dwk_r

            atomicaddsvec!(∑∇W, ∇w, pᵢ)

            #∑∇Wˣ = ∑∇W[1]
            #∑∇Wʸ = ∑∇W[2]
            #CUDA.@atomic ∑∇Wˣ[pᵢ] += ∇w[1]
            #CUDA.@atomic ∑∇Wʸ[pᵢ] += ∇w[2]

            atomicsubsvec!(∑∇W, ∇w, pⱼ)
            #CUDA.@atomic ∑∇Wˣ[pⱼ] -= ∇w[1]
            #CUDA.@atomic ∑∇Wʸ[pⱼ] -= ∇w[2]
        end
    end
    return nothing
end
# 3D
#=
function kernel_sph∑∇W!(∑∇W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, kernel, H⁻¹, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] >= 0 && ptype[pⱼ] >= 0 
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            u        = r * H⁻¹
            dwk_r = d𝒲(kernel, u, H⁻¹) / r
            ∇w    = (Δx[1] * dwk_r, Δx[2] * dwk_r, Δx[3] * dwk_r)
            ∑∇Wˣ = ∑∇W[1]
            ∑∇Wʸ = ∑∇W[2]
            ∑∇Wᶻ = ∑∇W[3]
            CUDA.@atomic ∑∇Wˣ[pᵢ] += ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pᵢ] += ∇w[2]
            CUDA.@atomic ∑∇Wᶻ[pᵢ] += ∇w[3]
            CUDA.@atomic ∑∇Wˣ[pⱼ] -= ∇w[1]
            CUDA.@atomic ∑∇Wʸ[pⱼ] -= ∇w[2]
            CUDA.@atomic ∑∇Wᶻ[pⱼ] -= ∇w[3]
        end
    end
    return nothing
end
=#
#####################################################################
# Thic can be used for all pair Δ calculations
# can reduce 40-45 μs per (100k pairs) for each equation
#####################################################################
#=
struct ParticlePair{T}
    Δxˣ::T
    Δxʸ::T
    Δvˣ::T
    Δvʸ::T
    #r²::T
    #r::T
    ρᵢ::T
    ρⱼ::T
    #function ParticlePair(Δx::Tuple{T, T}, Δv::Tuple{T, T}, ρᵢ::T, ρⱼ::T) where T
    #    new{T}(Δx, Δv, ρᵢ, ρⱼ)
    #end
end
function pairs_calk!(buff, pairs, ρ, v, points; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false kernel_pairs_calk!(buff, pairs, ρ, v, points) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff, pairs, ρ, v, points; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk!(buff, pairs, ρ, v, points) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        #pair  = pairs[index]
        #pᵢ    = pair[1]; pⱼ = pair[2]
        #xᵢ    = points[pᵢ]
        #xⱼ    = points[pⱼ]
        #Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
        #vᵢ    = v[pᵢ]
        #vⱼ    = v[pⱼ]
        #Δv    = (vᵢ[1] - vⱼ[1], vᵢ[2] - vⱼ[2])
        #buff[1][index] = Δx
        #buff[2][index] = Δv
        #buff[3][index] = ρ[pᵢ]
        #buff[4][index] = ρ[pⱼ]
        buff[1][index] = (1.2, 2.3)
        buff[2][index] = (4.5, 6.7)
        buff[3][index] = 0.1
        buff[4][index] = 0.7
    end
    return nothing
end
function pairs_calk2!(buff, pairs, ρ, v, points; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false maxregs=64 kernel_pairs_calk2!(buff, pairs, ρ, v, points) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    println(CUDA.registers(gpukernel), " ",maxThreads)
    println(CUDA.memory(gpukernel))
    CUDA.@sync gpukernel(buff, pairs, ρ, v, points; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk2!(buff, pairs, ρ, v, points::AbstractArray{Tuple{T, T}})  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while index <= length(buff)
        #pair  = pairs[index]
        #pᵢ    = pair[1]; pⱼ = pair[2]
        #xᵢ    = points[pᵢ]
        #xⱼ    = points[pⱼ]
        #Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
        #vᵢ    = v[pᵢ]
        #vⱼ    = v[pⱼ]
        #Δv    = (vᵢ[1] - vⱼ[1], vᵢ[2] - vⱼ[2])
        #r²    = Δx[1]^2 + Δx[2]^2
        #buff[index] = ParticlePair{T}(Δx, Δv, r², sqrt(r²), ρ[pᵢ], ρ[pⱼ])
        #buff[index] = ParticlePair{T}(xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2], vᵢ[1] - vⱼ[1],vᵢ[2] - vⱼ[2], ρ[pᵢ], ρ[pⱼ])
        buff[index] = ParticlePair{T}(0.1, 0.2, 0.3, 0.4, 0.9, 1.2)
        index += stride
    end
    return nothing
end
function pairs_calk3!(buff, pairs, ρ, v, points; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false kernel_pairs_calk3!(buff, pairs, ρ, v, points) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff, pairs, ρ, v, points; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk3!(buff, pairs, ρ, v, points) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        xᵢ    = points[pᵢ]
        xⱼ    = points[pⱼ]
        Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
        vᵢ    = v[pᵢ]
        vⱼ    = v[pⱼ]
        Δv    = (vᵢ[1] - vⱼ[1], vᵢ[2] - vⱼ[2])
        #r²    = Δx[1]^2 + Δx[2]^2
        #buff[index] = (Δx, Δv, r², sqrt(r²), ρ[pᵢ], ρ[pⱼ])
        buff[index] = (Δx[1], Δx[2], Δv[1],Δv[2], ρ[pᵢ], ρ[pⱼ])
    end
    return nothing
end
=#
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
    
    ∂ρ∂tDDT!(∑∂ρ∂t::CuArray{T}, pairs, ∇W, ρ, v, points, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024) 

Compute ∂ρ∂t - density derivative includind density diffusion. *Replace all values and update `∑∂ρ∂t`.*

```math
\\frac{\\partial \\rho_i}{\\partial t} = \\sum  m_j \\textbf{v}_{ij} \\cdot \\nabla_i W_{ij} + \\delta_{\\Phi} h c_0 \\sum \\Psi_{ij} \\cdot \\nabla_i W_{ij} \\frac{m_j}{\\rho_j}
```

```math
\\Psi_{ij} = 2 (\\rho_{ij}^T - \\rho_{ij}^H) \\frac{\\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}
```

```math
\\rho_{ij}^H = \\rho_0 \\left( \\sqrt[\\gamma]{\\frac{P_{ij}^H + 1}{C_b}} - 1\\right)
```

```math
P_{ij}^H = \\rho_0 g z_{ij}

```

``z_{ij}`` - vertical distance.

"""
function ∂ρ∂tDDT!(∑∂ρ∂t::CuArray{T}, pairs, ∇W, ρ, v, points, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024)  where T
     #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
    =#
    #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
    =#
    fill!(∑∂ρ∂t, zero(T))
    η²    = (0.1*h)*(0.1*h)
    γ⁻¹   = 1/γ
    DDTkh = 2 * h * δᵩ * c₀
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb
    if length(pairs) > length(first(∇W)) error("Length of pairs should be equal or less ∇W") end

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

    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ > 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0 # Only for liquid particled 
            getsvec
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)

            #xᵢ    = (points[1][pᵢ], points[2][pᵢ])
            #xⱼ    = (points[1][pⱼ], points[2][pⱼ])
            
            Δx    = xᵢ - xⱼ
            #Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = dot(Δx, Δx) # Δx[1]^2 + Δx[2]^2 

            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
   
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)  #(v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇Wᵢⱼ  = getsvec(∇W, index)
           
            dot3  = -dot(Δx, ∇Wᵢⱼ) #-(Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2]) #  - Δx ⋅ ∇Wᵢⱼ

            # as actual range at timestep Δt½  may be greateg  - some problems can be here
            # if 1 + DDTgz * Δx[2] < 0 || 1 - DDTgz * Δx[2] < 0 return nothing end
            
            m₀dot     = m₀ * dot(Δv, ∇Wᵢⱼ)  #(Δv[1] * ∇Wᵢⱼ[1] + Δv[2] * ∇Wᵢⱼ[2])  #  Δv ⋅ ∇Wᵢⱼ
            ∑∂ρ∂ti = ∑∂ρ∂tj = m₀dot

            if ptype[pᵢ] > 1
                drhopvp = ρ₀ * powfancy7th(1 + DDTgz * Δx[2], γ⁻¹, γ) - ρ₀ 
                #drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀
                visc_densi = DDTkh  * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
                delta_i    = visc_densi * dot3 * m₀ / ρⱼ
                ∑∂ρ∂ti    += delta_i #* (ptype[pᵢ] >= 1)
            end
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti 

            if ptype[pⱼ] > 1
                drhopvn = ρ₀ * powfancy7th(1 - DDTgz * Δx[2], γ⁻¹, γ) - ρ₀
                #drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
                visc_densi = DDTkh  * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
                delta_j    = visc_densi * dot3 * m₀ / ρᵢ
                ∑∂ρ∂tj    += delta_j #* (ptype[pⱼ] >= 1)
            end
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += ∑∂ρ∂tj  
                     
        end
        index += stride
    end
    return nothing
end
#=
function kernel_∂ρ∂tDDT!(∑∂ρ∂t, pairs, ∇W, ρ, v, points::NTuple{3, CuDeviceVector{T, 1}}, η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ > 0 && ptype[pᵢ] > 1 && ptype[pⱼ] > 1 # Only for liquid particled 
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ

            r²     = Δx[1]^2 + Δx[2]^2 + Δx[3]^2 
            ρᵢ     = ρ[pᵢ]
            ρⱼ     = ρ[pⱼ]
            Δv     = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2], v[pᵢ][3] - v[pⱼ][3])
            ∇Wᵢⱼ   = getsvec(∇W, index)
            dot3   = -(Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2] + Δx[3] * ∇Wᵢⱼ[3]) 
            m₀dot  = m₀ * (Δv[1] * ∇Wᵢⱼ[1] + Δv[2] * ∇Wᵢⱼ[2] + Δv[3] * ∇Wᵢⱼ[3]) 
            ∑∂ρ∂ti = ∑∂ρ∂tj = m₀dot

            drhopvp = ρ₀ * powfancy7th(1 + DDTgz * Δx[2], γ⁻¹, γ) - ρ₀ 
            visc_densi = DDTkh  * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
            delta_i    = visc_densi * dot3 * m₀ / ρⱼ
            ∑∂ρ∂ti    += delta_i
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti 

            drhopvn = ρ₀ * powfancy7th(1 - DDTgz * Δx[2], γ⁻¹, γ) - ρ₀
            visc_densi = DDTkh  * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            delta_j    = visc_densi * dot3 * m₀ / ρᵢ
            ∑∂ρ∂tj    += delta_j
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += ∑∂ρ∂tj             
        end
        index += stride
    end
    return nothing
end
=#
#####################################################################

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
"""
    pressure(ρ, c₀, γ, ρ₀)

Equation of State in Weakly-Compressible SPH

```math
P = c_0^2 \\rho_0 * \\left[  \\left( \\frac{\\rho}{\\rho_0} \\right)^{\\gamma}  \\right]
```
"""
function pressure(ρ, γ, ρ₀, P₀, ptype)
    #return  P₀ * ((ρ / ρ₀) ^ γ - 1) * (ptype < 1 && ρ < ρ₀)
    if ptype <= 1 && ρ < ρ₀
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
# 2D
function kernel_∂v∂t!(∑∂v∂t, ∇W, P, pairs, m₀, ρ, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] * ptype[pⱼ]  > 0 # for all particles (not for virtual)
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Pᵢ    = P[pᵢ]
            Pⱼ    = P[pⱼ]
            ∇Wᵢⱼ  = getsvec(∇W, index)
            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            ∂v∂t  = - m₀ * Pfac * ∇Wᵢⱼ
            #∑∂v∂tˣ = ∑∂v∂t[1]
            #∑∂v∂tʸ = ∑∂v∂t[2]   
            #CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  ∂v∂t[2]
            atomicaddsvec!(∑∂v∂t, ∂v∂t, pᵢ)


            #CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  ∂v∂t[2]
            atomicsubsvec!(∑∂v∂t, ∂v∂t, pⱼ)
        end
    end
    return nothing
end
# 3D
#=
function kernel_∂v∂t!(∑∂v∂t::NTuple{3, CuDeviceVector{T, 1}}, ∇W, P, pairs, m₀, ρ, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Pᵢ    = P[pᵢ]
            Pⱼ    = P[pⱼ]
            ∇Wᵢⱼ  = getsvec(∇W, index)
            Pfac  = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            ∂v∂t  = (- m₀ * Pfac * ∇Wᵢⱼ[1], - m₀ * Pfac * ∇Wᵢⱼ[2], - m₀ * Pfac * ∇Wᵢⱼ[3])
            ∑∂v∂tˣ = ∑∂v∂t[1]
            ∑∂v∂tʸ = ∑∂v∂t[2] 
            ∑∂v∂tᶻ = ∑∂v∂t[3]   
            CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂tᶻ[pᵢ] +=  ∂v∂t[3]
            CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂tᶻ[pⱼ] -=  ∂v∂t[3]
        end
    end
    return nothing
end
=#
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

\\\\

c_{ij} = c_0

\\\\

m_i = m_j = m_0

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
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r²    = dot(Δx, Δx) #Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)
            ρₘ     = (ρᵢ + ρⱼ) * 0.5
            ∇Wᵢⱼ   = getsvec(∇W, index)
            cond   = dot(Δv, Δx) #Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            if cond < 0
                Δμ   = h * cond / (r² + η²)
                ΔΠ   =  (-α * c₀ * Δμ) / ρₘ
                ΔΠm₀∇W = -ΔΠ * m₀ * ∇Wᵢⱼ
                atomicaddsvec!(∑∂v∂t, ΔΠm₀∇W, pᵢ)
                atomicsubsvec!(∑∂v∂t, ΔΠm₀∇W, pⱼ)

                #∑∂v∂tˣ = ∑∂v∂t[1]
                #∑∂v∂tʸ = ∑∂v∂t[2]   
                #CUDA.@atomic ∑∂v∂tˣ[pᵢ] += ΔΠm₀∇W[1]
                #CUDA.@atomic ∑∂v∂tʸ[pᵢ] += ΔΠm₀∇W[2]
                #CUDA.@atomic ∑∂v∂tˣ[pⱼ] -= ΔΠm₀∇W[1]
                #CUDA.@atomic ∑∂v∂tʸ[pⱼ] -= ΔΠm₀∇W[2]
            end
        end
    end
    return nothing
end
# 3D 
#=
function kernel_∂v∂t_av!(∑∂v∂t, ∇W, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, h, η², ρ, α, v, c₀, m₀, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ

            r²     = Δx[1]^2 + Δx[2]^2 + Δx[3]^2 
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)
            ρₘ     = (ρᵢ + ρⱼ) * 0.5
            ∇Wᵢⱼ   = getsvec(∇W, index)
            cond   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] +  Δv[3] * Δx[3] 
            if cond < 0
                Δμ   = h * cond / (r² + η²)
                ΔΠ   =  (-α * c₀ * Δμ) / ρₘ
                ΔΠm₀∇W = (-ΔΠ * m₀ * ∇Wᵢⱼ[1], -ΔΠ * m₀ * ∇Wᵢⱼ[2], -ΔΠ * m₀ * ∇Wᵢⱼ[3])
                ∑∂v∂tˣ = ∑∂v∂t[1]
                ∑∂v∂tʸ = ∑∂v∂t[2]  
                ∑∂v∂tᶻ = ∑∂v∂t[3]   
                CUDA.@atomic ∑∂v∂tˣ[pᵢ] += ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂v∂tʸ[pᵢ] += ΔΠm₀∇W[2]
                CUDA.@atomic ∑∂v∂tᶻ[pᵢ] += ΔΠm₀∇W[3]
                CUDA.@atomic ∑∂v∂tˣ[pⱼ] -= ΔΠm₀∇W[1]
                CUDA.@atomic ∑∂v∂tʸ[pⱼ] -= ΔΠm₀∇W[2]
                CUDA.@atomic ∑∂v∂tᶻ[pⱼ] -= ΔΠm₀∇W[3]
            end
        end
    end
    return nothing
end
=#
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
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r²    = dot(Δx, Δx)
            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)
            ∇Wᵢⱼ  = getsvec(∇W, index)
            𝜈term = 4𝜈 * m₀ * dot(Δx, ∇Wᵢⱼ) / ((ρᵢ + ρⱼ) * (r² + η²))  
            ∂v∂t  = 𝜈term * Δv

            atomicaddsvec!(∑∂v∂t, ∂v∂t, pᵢ)
            atomicsubsvec!(∑∂v∂t, ∂v∂t, pⱼ)

            #∑∂v∂tˣ = ∑∂v∂t[1]
            #∑∂v∂tʸ = ∑∂v∂t[2]   
            #CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  ∂v∂t[2]
            #CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  ∂v∂t[1]
            #CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  ∂v∂t[2]
        end
    end
    return nothing
end
# 3D 
#=
function kernel_∂v∂t_visc!(∑∂v∂t, ∇W, v, ρ, points::NTuple{3, CuDeviceVector{T, 1}}, pairs, η², m₀, 𝜈, ptype) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r²     = Δx[1]^2 + Δx[2]^2 + Δx[3]^2 
            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)
            ∇Wᵢⱼ  = getsvec(∇W, index)
            𝜈term = 4𝜈 * m₀ * (Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2] + Δx[3] * ∇Wᵢⱼ[3]) / ((ρᵢ + ρⱼ) * (r² + η²))  
            ∂v∂t  = (𝜈term * Δv[1], 𝜈term * Δv[2], 𝜈term * Δv[3])
            ∑∂v∂tˣ = ∑∂v∂t[1]
            ∑∂v∂tʸ = ∑∂v∂t[2]  
            ∑∂v∂tᶻ = ∑∂v∂t[3]    
            CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂tᶻ[pᵢ] +=  ∂v∂t[3]
            CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  ∂v∂t[1]
            CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  ∂v∂t[2]
            CUDA.@atomic ∑∂v∂tᶻ[pⱼ] -=  ∂v∂t[3]
        end
    end
    return nothing
end
=#
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
# 2D 
function kernel_∂v∂t_addgrav!(∑∂v∂t, gvec) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∑∂v∂t[1])
        subsvec!(∑∂v∂t, gvec, index)
        #∑∂v∂tˣ = ∑∂v∂t[1]
        #∑∂v∂tʸ = ∑∂v∂t[2]
        #∑∂v∂tˣ[index] -= gvec[1]
        #∑∂v∂tʸ[index] -= gvec[2]
    end
    return nothing
end
# 3D 
#=
function kernel_∂v∂t_addgrav!(∑∂v∂t::NTuple{3, CuDeviceVector{T, 1}}, gvec)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∑∂v∂t[1])
        ∑∂v∂tˣ = ∑∂v∂t[1]
        ∑∂v∂tʸ = ∑∂v∂t[2]
        ∑∂v∂tᶻ = ∑∂v∂t[3]
        ∑∂v∂tˣ[index] -= gvec[1]
        ∑∂v∂tʸ[index] -= gvec[2]
        ∑∂v∂tᶻ[index] -= gvec[3]
        
    end
    return nothing
end
=#
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
    if !(length(first(v)) == length(∑∂v∂t[1]) == length(ptype)) error("Wrong length") end
    gpukernel = @cuda launch = false kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(first(v))
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, ∑∂v∂t, Δt, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(v)) && ptype[index] > 1
        #val = getsvec(v, index) 
        #v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        ∂v∂t = getsvec(∑∂v∂t, index) 
        addsvec!(v, ∂v∂t * Δt, index)
        #v[index] = (val[1] + ∑∂v∂tˣ[index] * Δt, val[2] + ∑∂v∂tʸ[index] * Δt)
    end
    return nothing
end
#=
function kernel_update_vp∂v∂tΔt!(v, ∑∂v∂t::NTuple{3, CuDeviceVector{T, 1}}, Δt, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(v)) && ptype[index] > 1
        val = v[index]
        ∑∂v∂tˣ = ∑∂v∂t[1]
        ∑∂v∂tʸ = ∑∂v∂t[2] 
        ∑∂v∂tᶻ = ∑∂v∂t[3]
        v[index] = (val[1] + ∑∂v∂tˣ[index] * Δt, val[2] + ∑∂v∂tʸ[index] * Δt, val[3] + ∑∂v∂tᶻ[index] * Δt)
    end
    return nothing
end
=#
#####################################################################
"""
    update_xpvΔt!(x, v, Δt, ml) 

```math
\\textbf{r} = \\textbf{r} +  \\textbf{v} * \\Delta t
```

"""
function update_xpvΔt!(x, v, Δt) 
    if any(x->x != length(first(v)), length.(x)) error("Wrong vectors length...") end
    gpukernel = @cuda launch=false kernel_update_xpvΔt!(x, v, Δt) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(first(x))
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(x, v, Δt; threads = Tx, blocks = Bx)
end
function kernel_update_xpvΔt!(x, v, Δt)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))
        vval = getsvec(v,index)
        addsvec!(x, vval * Δt, index)
        #x[1][index] = x[1][index] + vval[1] * Δt
        #x[2][index] = x[2][index] + vval[2] * Δt
    end
    return nothing
end
#=
function kernel_update_xpvΔt!(x::NTuple{3, CuDeviceVector{T, 1}}, v, Δt)  where T 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))
        vval = v[index]
        x[1][index] = x[1][index] + vval[1] * Δt
        x[2][index] = x[2][index] + vval[2] * Δt
        x[3][index] = x[3][index] + vval[3] * Δt
    end
    return nothing
end
=#
#####################################################################
"""
    
    symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 

Symplectic Position Verlet scheme.

* Parshikov et al, 2000
* Leimkuhler and Matthews, 2016

"""
function symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) 
    if any(x->x != length(first(v)), length.(x)) error("Wrong vectors length...") end
    gpukernel = @cuda launch=false kernel_symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, cΔx, ρ₀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(first(x))
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype; threads = Tx, blocks = Bx)
end
# 2D 
function kernel_symplectic_update!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)

        if ρval < ρ₀ && ptype[index] <= 1 ρval = ρ₀ end

        ρΔt½[index] = ρval
        ρ[index]    = ρval
        vval        = getsvec(v, index)

        #∑∂v∂tˣ      = ∑∂v∂t[1]
        #∑∂v∂tʸ      = ∑∂v∂t[2] 
        ∂v∂t        = getsvec(∑∂v∂t, index)
        ml          = ifelse(ptype[index] > 1, 1.0, 0.0)
        nval        = vval + ∂v∂t * Δt * ml 

        setsvec!(vΔt½, nval, index)
        setsvec!(v, nval, index)

        xval           = getsvec(x, index) 
        #Δxˣ, Δxʸ       = (vval[1] + nval[1]) * 0.5  * Δt, (vval[2] + nval[2]) * 0.5  * Δt
        Δx             = (vval + nval) * 0.5  * Δt
        #cΔx[1][index] += Δxˣ
        #cΔx[2][index] += Δxʸ
        addsvec!(cΔx, Δx, index)

        xval           = xval + Δx#(xval[1] + Δxˣ, xval[2] + Δxʸ)

        #xΔt½[1][index] = xval[1]
        #xΔt½[2][index] = xval[2]
        setsvec!(xΔt½, xval, index)


        #x[1][index] = xval[1]
        #x[2][index] = xval[2]
        setsvec!(x, xval, index)
    end
    return nothing
end
# 3D 
#=
function kernel_symplectic_update!(ρ, ρΔt½, v, vΔt½, x::NTuple{3, CuDeviceVector{T, 1}}, xΔt½, ∑∂ρ∂t, ∑∂v∂t, Δt, cΔx, ρ₀, ptype)  where T 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))

        epsi       = -(∑∂ρ∂t[index] / ρΔt½[index]) * Δt
        ρval       = ρ[index]  * (2 - epsi)/(2 + epsi)
        if ρval < ρ₀ && ptype[index] < 0 ρval = ρ₀ end

        ρΔt½[index] = ρval
        ρ[index]    = ρval
        vval        = getsvec(v, index)

        ∑∂v∂tˣ      = ∑∂v∂t[1]
        ∑∂v∂tʸ      = ∑∂v∂t[2]
        ∑∂v∂tᶻ      = ∑∂v∂t[3]
        ml          = ifelse(ptype[index] >= 1, 1.0, 0.0)
        nval        = (vval[1] +  ∑∂v∂tˣ[index] * Δt * ml, vval[2]  + ∑∂v∂tʸ[index] * Δt * ml, vval[3]  + ∑∂v∂tᶻ[index] * Δt * ml)

        setsvec!(vΔt½, nval, index)
        setsvec!(v, nval, index)

        xval           = (x[1][index], x[2][index], x[3][index])
        Δxˣ, Δxʸ, Δxᶻ       = (vval[1] + nval[1]) * 0.5  * Δt, (vval[2] + nval[2]) * 0.5  * Δt, (vval[3] + nval[3]) * 0.5  * Δt
        cΔx[1][index] += Δxˣ
        cΔx[2][index] += Δxʸ
        cΔx[3][index] += Δxᶻ
        xval           = (xval[1] + Δxˣ, xval[2] + Δxʸ, xval[3] + Δxᶻ)

        xΔt½[1][index] = xval[1]
        xΔt½[2][index] = xval[2]
        xΔt½[3][index] = xval[3]

        x[1][index] = xval[1]
        x[2][index] = xval[2]
        x[3][index] = xval[3]

    end
    return nothing
end
=#
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
        vp = getsvec(v, index)
        pp = getsvec(points, index)
        #buf[index] = abs(h * (vp[1] * pp[1] + vp[2] * pp[2]) / (pp[1]^2 + pp[2]^2 + η²))
        buf[index] = abs(h * dot(vp, pp) / (dot(pp, pp) + η²))
    end
    return nothing
end
#=
function kernel_Δt_stepping!(buf, v, points::NTuple{3, CuDeviceVector{T, 1}}, h, η²)   where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        vp = getsvec(v, index)
        pp = (points[1][index], points[2][index], points[3][index])
        buf[index] = abs(h * (vp[1] * pp[1] + vp[2] * pp[2] + vp[3] * pp[3]) / (pp[1]^2 + pp[2]^2 + pp[3]^2 + η²))
    end
    return nothing
end
=#
function kernel_Δt_stepping_norm!(buf, a) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        avec       = getsvec(a, index)
        buf[index] =  sqrt(dot(avec, avec)) 
    end
    return nothing
end
#=
function kernel_Δt_stepping_norm!(buf, a::NTuple{3, CuDeviceVector{T, 1}})  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        buf[index] =  sqrt(a[1][index]^2 + a[2][index]^2 + a[3][index]^2) 
    end
    return nothing
end
=#
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
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
           
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            if r < 2h
                scos   = s * cos(1.5π * r / 2h)/ (r + (0.1*h))
                ∂v∂tpF = scos * Δx / m₀
                atomicaddsvec!(∑∂v∂t, ∂v∂tpF, pᵢ)
                atomicsubsvec!(∑∂v∂t, ∂v∂tpF, pⱼ)
                #∑∂v∂tˣ = ∑∂v∂t[1]
                #∑∂v∂tʸ = ∑∂v∂t[2] 
                #CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  scos * Δx[1] / m₀
                #CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  scos * Δx[2] / m₀
                #CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  scos * Δx[1] / m₀
                #CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  scos * Δx[2] / m₀
            end

        end
    end
    return nothing
end
# 3D 
#=
function kernel_∂v∂tpF!(∑∂v∂t, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, s, h, m₀, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
           
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r     = norm(Δx) 
            if r < 2h
                scos = s * cos(1.5π * r / 2h)/ (r + (0.1*h))
                ∑∂v∂tˣ = ∑∂v∂t[1]
                ∑∂v∂tʸ = ∑∂v∂t[2] 
                ∑∂v∂tᶻ = ∑∂v∂t[3] 
                CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  scos * Δx[1] / m₀
                CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  scos * Δx[2] / m₀
                CUDA.@atomic ∑∂v∂tᶻ[pᵢ] +=  scos * Δx[3] / m₀

                CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  scos * Δx[1] / m₀
                CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  scos * Δx[2] / m₀
                CUDA.@atomic ∑∂v∂tᶻ[pⱼ] -=  scos * Δx[3] / m₀
            end

        end
    end
    return nothing
end
=#
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

```

```math
(v_{ij}^{coll} , \\quad \\phi_{ij}) = \\begin{cases} (\\frac{\\textbf{v}_{ij}\\cdot \\textbf{r}_{ij}}{r_{ij}^2 + \\eta^2}\\textbf{r}_{ji}, \\quad 0) & \\textbf{v}_{ij}\\cdot \\textbf{r}_{ij} < 0 \\\\ (0, \\quad 1) &  otherwise \\end{cases}

```math
p_{ij}^b = \\tilde{p}_{ij} \\chi_{ij} 
```


```math
\\tilde{p}_{ij} = max(min(\\lambda |p_i + p_j|, \\lambda p_{max}), p_{min})
```

```math
\\chi_{ij}  = \\sqrt{\\frac{\\omega({r}_{ij}, l_0)}{\\omega(l_0/2, l_0)}}

```

```math
k_{ij} =  \\begin{cases} \\chi_{ij} & 0.5 \\le {r}_{ij}/l_0 < 1 \\\\ 1 & {r}_{ij}/l_0 < 0.5 \\end{cases}
```

Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf

"""
function dpcreg!(∑Δvdpc, v, ρ::CuArray{T}, P::CuArray{T}, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim, ptype) where T

    for vec in ∑Δvdpc fill!(vec, zero(T)) end

    l₀⁻¹     = 1 / l₀  
    wh⁻¹     = 1 / 𝒲(sphkernel, 0.5, l₀⁻¹)
    gpukernel = @cuda launch=false kernel_dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim, ptype; threads = Tx, blocks = Bx)
end
# 2D 
function kernel_dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            η²    = (0.1 * l₀) * (0.1 * l₀)
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ

            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)  # (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])
            r²    = dot(Δx, Δx)# Δx[1]^2 + Δx[2]^2 
            r     = sqrt(r²) 
            u     = r * l₀⁻¹
            w     = 𝒲(sphkernel, u, l₀⁻¹)

            χ     = sqrt(w * wh⁻¹)

            k     = ifelse(u < dpckernlim, 1.0, χ)

            Pᵇ    = χ * max(min(λ * abs(P[pᵢ] + P[pⱼ]), λ * Pmax), Pmin)

            vr   = dot(Δv, Δx) # Δv[1] * Δx[1] +  Δv[2] * Δx[2] 

            if vr < 0
                # Δvdpc = ∑ k * 2mⱼ / (mᵢ + mⱼ) * vᶜ   | mⱼ = mᵢ |  => Δvdpc = ∑ k * vᶜ
                vrdr    = vr / (r² + η²)
                vᶜ      = -vrdr * Δx #(-vrdr * Δx[1],  -vrdr * Δx[2])
                Δvdpc   = k * vᶜ #(k * vᶜ[1],  k * vᶜ[2])
            else
                # Δvdpc = Δt / ρᵢ * ∑ 2Vᵢ / (Vᵢ + Vⱼ) * Pᵇ / (r² + η²) * Δx
                # V = m / ρ
                # Δvdpc = Δt * ∑ 2 / (ρᵢ + ρⱼ) * Pᵇ / (r² + η²) * Δx
                tvar = 2Δt* Pᵇ / ((ρᵢ + ρⱼ) * (r² + η²))
                Δvdpc = tvar * Δx # (tvar * Δx[1], tvar * Δx[2])
            end
            atomicaddsvec!(∑Δvdpc, Δvdpc, pᵢ)
            atomicsubsvec!(∑Δvdpc, Δvdpc, pⱼ)
            
            #∑Δvdpcˣ = ∑Δvdpc[1]
            #∑Δvdpcʸ = ∑Δvdpc[2]   
            #CUDA.@atomic ∑Δvdpcˣ[pᵢ] +=  Δvdpc[1]
            #CUDA.@atomic ∑Δvdpcʸ[pᵢ] +=  Δvdpc[2]
            #CUDA.@atomic ∑Δvdpcˣ[pⱼ] -=  Δvdpc[1]
            #CUDA.@atomic ∑Δvdpcʸ[pⱼ] -=  Δvdpc[2]
        end
    end
    return nothing
end
# 3D
#=
function kernel_dpcreg!(∑Δvdpc, v, ρ, P, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, sphkernel, wh⁻¹, l₀, l₀⁻¹, Pmin, Pmax, Δt, λ, dpckernlim, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 1 && ptype[pⱼ] > 1
            η²    = (0.1 * l₀) * (0.1 * l₀)
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ

            r²     = Δx[1]^2 + Δx[2]^2 + Δx[3]^2
            r     = sqrt(r²) 
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2], v[pᵢ][3] - v[pⱼ][3])
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            u     = r * l₀⁻¹
            w     = 𝒲(sphkernel, u, l₀⁻¹)
            χ     = sqrt(w * wh⁻¹)
            k     = ifelse(u < dpckernlim, 1.0, χ)
            Pᵇ    = χ * max(min(λ * abs(P[pᵢ] + P[pⱼ]), λ * Pmax), Pmin)
            vr   = Δv[1] * Δx[1] +  Δv[2] * Δx[2] +  Δv[3] * Δx[3] 

            if vr < 0
                vrdr    = vr / (r² + η²)
                vᶜ      = (-vrdr * Δx[1],  -vrdr * Δx[2],  -vrdr * Δx[3])
                Δvdpc   = (k * vᶜ[1],  k * vᶜ[2],  k * vᶜ[3])
            else
                tvar = 2Δt* Pᵇ / ((ρᵢ + ρⱼ) * (r² + η²))
                Δvdpc = (tvar * Δx[1], tvar * Δx[2], tvar * Δx[3])
            end
            ∑Δvdpcˣ = ∑Δvdpc[1]
            ∑Δvdpcʸ = ∑Δvdpc[2]
            ∑Δvdpcᶻ = ∑Δvdpc[3]
            CUDA.@atomic ∑Δvdpcˣ[pᵢ] +=  Δvdpc[1]
            CUDA.@atomic ∑Δvdpcʸ[pᵢ] +=  Δvdpc[2]
            CUDA.@atomic ∑Δvdpcᶻ[pᵢ] +=  Δvdpc[3]
            CUDA.@atomic ∑Δvdpcˣ[pⱼ] -=  Δvdpc[1]
            CUDA.@atomic ∑Δvdpcʸ[pⱼ] -=  Δvdpc[2]
            CUDA.@atomic ∑Δvdpcᶻ[pⱼ] -=  Δvdpc[3]
        end
    end
    return nothing
end 
=#
"""
    update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 

Update velocity and position.
"""
function update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    if any(x->x != length(first(v)), length.(x)) error("Wrong vectors length...") end
    gpukernel = @cuda launch=false kernel_update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(first(x))
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, x, ∑Δvdpc, Δt, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_dpcreg!(v, x, ∑Δvdpc, Δt, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))
        if ptype[index] > 1
            #xval = getsvec(x, index) #(x[1][index], x[2][index])
            #vval = getsvec(v, index)
            dpcval = getsvec(∑Δvdpc, index) #(∑Δvdpc[1][index], ∑Δvdpc[2][index])

            addsvec!(v, dpcval, index)
            #v[index] = (vval[1] + dpcval[1], vval[2] + dpcval[2])
            addsvec!(x, dpcval * Δt, index)
            #x[1][index], x[2][index] = (xval[1] + dpcval[1] * Δt, xval[2] + dpcval[2] * Δt)
        end
    end
    return nothing
end
#=
function kernel_update_dpcreg!(v, x::NTuple{3, CuDeviceVector{T, 1}}, ∑Δvdpc, Δt, ptype) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(x))
        if ptype[index] > 1
            xval = (x[1][index], x[2][index], x[3][index])
            vval = v[index]
            dpcval = (∑Δvdpc[1][index], ∑Δvdpc[2][index], ∑Δvdpc[3][index])

            v[index] = (vval[1] + dpcval[1], vval[2] + dpcval[2], vval[3] + dpcval[3])
            x[1][index], x[2][index], x[3][index] = (xval[1] + dpcval[1] * Δt, xval[2] + dpcval[2] * Δt, xval[3] + dpcval[3] * Δt)
        end
    end
    return nothing
end
=#
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

    for vec in ∑ρcspm fill!(vec, zero(T)) end

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
        if pᵢ != 0 && ptype[pᵢ] > 0 && ptype[pⱼ] > 0 # for liquid and boundary cover
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
function xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀, ptype)

    for vec in ∑Δvxsph fill!(vec, zero(eltype(vec))) end

    gpukernel = @cuda launch=false kernel_xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀, ptype; threads = Tx, blocks = Bx)
end
function kernel_xsphcorr!(∑Δvxsph, pairs, W, ρ, v, m₀, 𝜀, ptype)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ptype[pᵢ] > 1 && ptype[pⱼ] > 1
            Δv    = getsvec(v, pᵢ) - getsvec(v, pⱼ)
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            xsph  = 2m₀ * 𝜀 * W[index] / (ρᵢ + ρⱼ)
            xsphv = Δv * xsph 
            atomicsubsvec!(∑Δvxsph, xsphv, pᵢ)
            atomicaddsvec!(∑Δvxsph, xsphv, pⱼ)
            #∑Δvxsphˣ = ∑Δvxsph[1]
            #∑Δvxsphʸ = ∑Δvxsph[2]
            #CUDA.@atomic ∑Δvxsphˣ[pᵢ] -=  xsphv[1]
            #CUDA.@atomic ∑Δvxsphʸ[pᵢ] -=  xsphv[2]
            #CUDA.@atomic ∑Δvxsphˣ[pⱼ] +=  xsphv[1]
            #CUDA.@atomic ∑Δvxsphʸ[pⱼ] +=  xsphv[2]
        end
    end
    return nothing
end
# 3D 
#=
function kernel_xsphcorr!(∑Δvxsph::NTuple{3, CuDeviceVector{T, 1}}, pairs, W, ρ, v, m₀, 𝜀)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2], v[pᵢ][3] - v[pⱼ][3])
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]
            xsph  = 2m₀ * 𝜀 * W[index] / (ρᵢ + ρⱼ)
            xsphv = (xsph * Δv[1], xsph * Δv[2], xsph * Δv[3])
            ∑Δvxsphˣ = ∑Δvxsph[1]
            ∑Δvxsphʸ = ∑Δvxsph[2]
            ∑Δvxsphᶻ = ∑Δvxsph[3]
            CUDA.@atomic ∑Δvxsphˣ[pᵢ] -=  xsphv[1]
            CUDA.@atomic ∑Δvxsphʸ[pᵢ] -=  xsphv[2]
            CUDA.@atomic ∑Δvxsphᶻ[pᵢ] -=  xsphv[3]

            CUDA.@atomic ∑Δvxsphˣ[pⱼ] +=  xsphv[1]
            CUDA.@atomic ∑Δvxsphʸ[pⱼ] +=  xsphv[2]
            CUDA.@atomic ∑Δvxsphᶻ[pⱼ] +=  xsphv[3]
        end
    end
    return nothing
end
=#
"""
    update_xsphcorr!(v, ∑Δvxsph, ptype) 

Update velocity.
"""
function update_xsphcorr!(v, ∑Δvxsph, ptype) 
    if length(first(v)) != length(ptype) error("length error") end
    gpukernel = @cuda launch=false kernel_update_xsphcorr!(v, ∑Δvxsph, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(first(v))
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(v, ∑Δvxsph, ptype; threads = Tx, blocks = Bx)
end
function kernel_update_xsphcorr!(v, ∑Δvxsph, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(first(v))
        if ptype[index] > 1
            #vval = v[index]
            #xsph = (∑Δvxsph[1][index], ∑Δvxsph[2][index])
            #v[index] = (vval[1] + xsph[1], vval[2] + xsph[2])
            addsvec!(v, getsvec(∑Δvxsph, index), index)
        end
    end
    return nothing
end
#=
function kernel_update_xsphcorr!(v, ∑Δvxsph::NTuple{3, CuDeviceVector{T, 1}}, ptype) where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(v)
        if ptype[index] > 1
            vval = v[index]
            xsph = (∑Δvxsph[1][index], ∑Δvxsph[2][index], ∑Δvxsph[3][index])
            v[index] = (vval[1] + xsph[1], vval[2] + xsph[2], vval[3] + xsph[3])
        end
    end
    return nothing
end
=#
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
# 2D 
function kernel_fbmolforce!(∑∂v∂t, pairs, points, d, r₀, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    η² = 0.01*r₀^2
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ((ptype[pᵢ] > 0) ⊻ (ptype[pⱼ] > 0))
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r²    = dot(Δx, Δx) 
            r     = sqrt(r²)
            if r < r₀
                Fc    = d * ((r₀ / (r + η²))^12 - (r₀ / (r + η²))^4) / (r² + η²) 
                F     = Δx * Fc 
                atomicaddsvec!(∑∂v∂t, F, pᵢ)
                atomicsubsvec!(∑∂v∂t, F, pⱼ)

                #∑∂v∂tˣ = ∑∂v∂t[1]
                #∑∂v∂tʸ = ∑∂v∂t[2] 
                #CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  F[1]
                #CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  F[2]
                #CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  F[1]
                #CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  F[2]
            end
        end
    end
    return nothing
end
# 3D 
#=
function kernel_fbmolforce!(∑∂v∂t, pairs, points::NTuple{3, CuDeviceVector{T, 1}}, d, r₀, ptype)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ != 0 && ((ptype[pᵢ] > 0) ⊻ (ptype[pⱼ] > 0))
            xᵢ    = getsvec(points, pᵢ)
            xⱼ    = getsvec(points, pⱼ)
            Δx    = xᵢ -  xⱼ
            r²     = Δx[1]^2 + Δx[2]^2 + Δx[3]^2
            r     = sqrt(r²) 
            if r < r₀
                Fc    = d * ((r₀ / r)^12 - (r₀ / r)^4) / r² 
                F     = (Δx[1] * Fc, Δx[2] * Fc, Δx[3] * Fc)
                
                ∑∂v∂tˣ = ∑∂v∂t[1]
                ∑∂v∂tʸ = ∑∂v∂t[2]
                ∑∂v∂tᶻ = ∑∂v∂t[3]

                CUDA.@atomic ∑∂v∂tˣ[pᵢ] +=  F[1]
                CUDA.@atomic ∑∂v∂tʸ[pᵢ] +=  F[2]
                CUDA.@atomic ∑∂v∂tᶻ[pᵢ] +=  F[3]
                
                CUDA.@atomic ∑∂v∂tˣ[pⱼ] -=  F[1]
                CUDA.@atomic ∑∂v∂tʸ[pⱼ] -=  F[2]
                CUDA.@atomic ∑∂v∂tᶻ[pⱼ] -=  F[3]
            end
        end
    end
    return nothing
end
=#
function fbmolforce(d, r₀, r)
    ifelse(r < r₀, d * ((r₀ / r)^12 - (r₀ / r)^4) / r^2, 0.0)
end