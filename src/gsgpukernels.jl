function gskernel_∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)
            u     = d * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic sumW[pᵢ] += w
            CUDA.@atomic sumW[pⱼ] += w
        end
        index += stride
    end
    return nothing
end
"""

    ∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 

Compute ∑W for each particles pair in list.
"""
function gs∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false gskernel_∑W_2d!(sumW, pairs, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = min(config.blocks, cld(Nx, Tx))
    CUDA.@sync gpukernel(sumW, pairs, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end


#####################################################################


function gskernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    while index <= size(∑∂v∂t, 1)
        val = v[index]
        v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        index += stride
    end
    return nothing
end
"""
    update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 


"""
function gsupdate_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    if length(v) != size(∑∂v∂t, 1) error("Wrong length") end
    gpukernel = @cuda launch=false gskernel_update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = size(∑∂v∂t, 1)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = min(config.blocks, cld(Nx, Tx))
    CUDA.@sync gpukernel(v, ∑∂v∂t, Δt, ml; threads = Tx, blocks = Bx)
end