function kernel_∑W∑∇W_2d!(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)
            u     = d * H⁻¹
            w     = 𝒲(sphkernel, u, H⁻¹)
            CUDA.@atomic ∑W[pᵢ] += w
            CUDA.@atomic ∑W[pⱼ] += w

            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]

            dwk_r = d𝒲(sphkernel, u, H⁻¹) / d
            ∇w    = ((xᵢ[1] - xⱼ[1]) * dwk_r, (xᵢ[2] - xⱼ[2]) * dwk_r)
            CUDA.@atomic ∑∇W[pᵢ, 1] += ∇w[1]
            CUDA.@atomic ∑∇W[pᵢ, 2] += ∇w[2]
            CUDA.@atomic ∑∇W[pⱼ, 1] -= ∇w[1]
            CUDA.@atomic ∑∇W[pⱼ, 2] -= ∇w[2]
            ∇Wₙ[index] = ∇w

        end
    end
    return nothing
end
"""

    ∑W∑∇W_2d!(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹) 

Compute ∑W ans gradient for each particles pair in list.
"""
function ∑W∑∇W_2d!(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹) 
    gpukernel = @cuda launch=false kernel_∑W∑∇W_2d!(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹; threads = Tx, blocks = Bx)
end




#####################################################################
function kernel_update_Δt!(ρ, v, x, ∑∂ρ∂t, ∑∂v∂t, Δt, ρ₀, isboundary, ml) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(ρ)
        # update ρ
        @inbounds val = ρ[index] + ∑∂ρ∂t[index] * Δt
        if val < ρ₀ && isboundary[index] val = ρ₀ end
        @inbounds ρ[index] = val
        # update v
        @inbounds val = v[index]
        @inbounds val = v[index] = (val[1] + ∑∂v∂t[index, 1] * Δt * ml[index], val[2] + ∑∂v∂t[index, 2] * Δt * ml[index])
        # update x
        @inbounds xval = x[index]
        @inbounds x[index] = (xval[1] + val[1] * Δt * ml[index], xval[2] + val[2] * Δt * ml[index])
    end
    return nothing
end
"""
    update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary) 

Update ρ, v, x at timestep Δt and derivatives: ∑∂ρ∂t, ∑∂v∂t.
"""
function update_Δt!(ρ, v, x, ∑∂ρ∂t, ∑∂v∂t, Δt, ρ₀, isboundary, ml) 
    if !(length(ρ) == length(v) == length(x)== size(∑∂ρ∂t, 1) == size(∑∂v∂t, 1) == length(isboundary) == length(ml)) error("Wrong length") end

    gpukernel = @cuda launch=false kernel_update_Δt!(ρ, v, x, ∑∂ρ∂t, ∑∂v∂t, Δt, ρ₀, isboundary, ml) 
    config = launch_configuration(gpukernel.fun)
    Nx = size(∑∂ρ∂t, 1)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(ρ, v, x, ∑∂ρ∂t, ∑∂v∂t, Δt, ρ₀, isboundary, ml; threads = Tx, blocks = Bx)
end

#####################################################################
#####################################################################


function _stepsolve!(prob::SPHProblem, n::Int, ::Effective; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))
    if timestepping && timelims[1] > timelims[1] error("timelims[1] should be < timelims[2]") end
    for iter = 1:n

        update!(prob.system)
        x     = prob.system.points
        pairs = neighborlist(prob.system)

        fill!(prob.∑W, zero(Float64))
        fill!(prob.∑∂ρ∂t, zero(Float64))
        fill!(prob.∑∂Π∂t, zero(Float64))
        fill!(prob.∑∂v∂t, zero(Float64))

        if length(prob.∇Wₙ) != length(pairs)
            CUDA.unsafe_free!(prob.∇Wₙ)
            prob.∇Wₙ =  CUDA.fill((zero(Float64), zero(Float64)), length(pairs)) # DIM = 2
        end
        # kernels sum for each cell & kernels gradient  for each cell (∑∇W) and value for each pair (∇Wₙ)
        ∑W∑∇W_2d!(prob.∑W, prob.∑∇W, prob.∇Wₙ, pairs, x, prob.sphkernel, prob.H⁻¹) 

        # density derivative with density diffusion
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, prob.∇Wₙ, pairs, x, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 
        # artificial viscosity
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρ, prob.c₀, prob.γ, prob.ρ₀) 
        # add gravity and artificial viscosity 
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t,  gravvec(prob.g, prob.dim), prob.gf) 
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, x, prob.s, prob.h, prob.m₀, prob.isboundary) 
        end
        
        # following steps (update_ρ!, update_vp∂v∂tΔt!, update_xpvΔt!) can be done in one kernel 
        # calc ρ at Δt½
        # calc v at Δt½
        # calc x at Δt½
        update_Δt!(prob.ρΔt½, prob.vΔt½, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt * 0.5, prob.ρ₀, prob.isboundary, prob.ml) 

        # set derivative to zero for Δt½ calc
        fill!(prob.∑∂ρ∂t, zero(Float64))
        fill!(prob.∑∂Π∂t, zero(Float64))
        fill!(prob.∑∂v∂t, zero(Float64))
        # density derivative with density diffusion at  xΔt½ 
        ∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 
        # artificial viscosity at xΔt½ 
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        # momentum equation at ρΔt½
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρΔt½, prob.c₀, prob.γ, prob.ρ₀) 
        # add gravity and artificial viscosity 
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t, gravvec(prob.g, prob.dim), prob.gf)
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, prob.xΔt½, prob.s, prob.h, prob.m₀, prob.isboundary) 
        end
        # update all with symplectic position Verlet scheme
        update_all!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.ρ₀, prob.isboundary, prob.ml)
   
        prob.etime += prob.Δt

        if timestepping
            prob.Δt = Δt_stepping(prob.buf, prob.∑∂v∂t, prob.v, x, prob.c₀, prob.h, prob.CFL, timelims)
        end
 
    end
end
