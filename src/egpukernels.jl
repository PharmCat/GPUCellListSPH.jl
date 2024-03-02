#####################################################################
#####################################################################


function _stepsolve!(prob::SPHProblem, n::Int, ::Effective; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))
    if timestepping && timelims[1] > timelims[1] error("timelims[1] should be < timelims[2]") end

    x              = prob.system.points
    pairs          = neighborlist(prob.system)
    skipupdate     = false
    updaten        = 0
    skipupdaten    = 0
    maxcΔx         = 0.0
    maxcΔxout      = 0.0
    cspmcorrn      = 0
    dpckernlim = find_zero(x-> 1.0 - 𝒲(prob.sphkernel, x, 1.0), 0.5)

    for iter = 1:n
        cspmcorrn       += 1
        if skipupdate 
            skipupdaten += 1
        else
            update!(prob.system)
            x           = prob.system.points
            pairs       = neighborlistview(prob.system)
            #sort!(pairs, by = first)
            for a in prob.cΔx fill!(a, zero(T)) end
            skipupdate  = true
            updaten += 1 
        end


        fill!(prob.∑∂v∂t[1], zero(T))
        fill!(prob.∑∂v∂t[2], zero(T))


        if length(prob.∇W) != length(pairs)
            CUDA.unsafe_free!(prob.∇W)
            CUDA.unsafe_free!(prob.W)
            prob.∇W =  CUDA.fill((zero(T), zero(Float64)), length(pairs)) # DIM = 2
            prob.W =  CUDA.fill(zero(T), length(pairs))
        end
        # kernels for each pair
        W_2d!(prob.W, pairs, x, prob.H⁻¹, prob.sphkernel)
        # kernels gradientfor each pair
        ∇W_2d!(prob.∇W, pairs, x, prob.H⁻¹, prob.sphkernel)
        # density derivative with density diffusion
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, pairs, prob.∇W, prob.ρ, prob.v, x, prob.h, prob.m₀, prob.ρ₀, prob.c₀, prob.γ, prob.g, prob.δᵩ, prob.ptype; minthreads = 256) 
        #  pressure
        pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇W, prob.P, pairs,  prob.m₀, prob.ρ, prob.ptype) 
        # add artificial viscosity
        ∂v∂t_av!(prob.∑∂v∂t, prob.∇W, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀, prob.ptype)
        # laminar shear stresse
        if prob.𝜈 > 0
            ∂v∂t_visc!(prob.∑∂v∂t, prob.∇W, prob.v, prob.ρ, x, pairs, prob.h, prob.m₀, prob.𝜈, prob.ptype)
        end
        # add gravity 
        ∂v∂t_addgrav!(prob.∑∂v∂t, gravvec(prob.g, prob.dim)) 
        #  Boundary forces
        fbmolforce!(prob.∑∂v∂t, pairs, x, 0.4, 2 * prob.dx, prob.ptype)
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, x, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        
        # following steps (update_ρ!, update_vp∂v∂tΔt!, update_xpvΔt!) can be done in one kernel 
        # calc ρ at Δt½
        update_ρp∂ρ∂tΔt!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.ptype)
        # calc v at Δt½
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ptype) 
        # calc x at Δt½
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5)

        # set derivative to zero for Δt½ calc

        fill!(prob.∑∂v∂t[1], zero(T))
        fill!(prob.∑∂v∂t[2], zero(T))
        

        # density derivative with density diffusion at  xΔt½ 
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, pairs, prob.∇W, prob.ρΔt½, prob.vΔt½, prob.xΔt½, prob.h, prob.m₀, prob.ρ₀, prob.c₀, prob.γ, prob.g, prob.δᵩ, prob.ptype; minthreads = 256) 
        #  pressure
        pressure!(prob.P, prob.ρΔt½, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t, prob.∇W, prob.P, pairs,  prob.m₀, prob.ρΔt½, prob.ptype)
        # add artificial viscosity at xΔt½ 
        ∂v∂t_av!(prob.∑∂v∂t, prob.∇W, pairs, prob.xΔt½, prob.h, prob.ρΔt½, prob.α, prob.vΔt½, prob.c₀, prob.m₀, prob.ptype)
        # laminar shear stresse
        if prob.𝜈 > 0
            ∂v∂t_visc!(prob.∑∂v∂t, prob.∇W, prob.vΔt½, prob.ρΔt½, prob.xΔt½, pairs, prob.h, prob.m₀, prob.𝜈, prob.ptype)
        end
        # add gravity 
        ∂v∂t_addgrav!(prob.∑∂v∂t,gravvec(prob.g, prob.dim))
        #  Boundary forces
        fbmolforce!(prob.∑∂v∂t, pairs, x, 0.4, 2 * prob.dx, prob.ptype)
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, prob.xΔt½, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        # update all with symplectic position Verlet scheme
        symplectic_update!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.cΔx, prob.ρ₀, prob.ptype)
        
        # Dynamic Particle Collision (DPC) 
        if prob.dpc_l₀ > 0
            #  pressure
            pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
            dpcreg!(prob.buf2, prob.v, prob.ρ, prob.P, pairs, x, prob.sphkernel, prob.dpc_l₀, prob.dpc_pmin, prob.dpc_pmax, prob.Δt, prob.dpc_λ, dpckernlim)  
            update_dpcreg!(prob.v, x, prob.buf2, prob.Δt, prob.ptype)
        end

        # XSPH correction.
        if prob.xsph_𝜀 > 0
            xsphcorr!(prob.buf2, prob.pairs, prob.W, prob.ρ, prob.v, prob.m₀, prob.𝜀)
            update_xsphcorr!(prob.v, prob.buf2, prob.ptype) 
        end


        # Density Renormalisation every 15 timesteps
        if cspmcorrn == 15
            cspmcorr!(prob.buf2, prob.W, prob.ρ, prob.m₀, pairs, prob.ptype)
            cspmcorrn = 0
        end


        maxcΔx = maximum(maximum.(abs, prob.cΔx))
        if maxcΔx > 0.9 * prob.nui  
            skipupdate = false 
        end
        maxcΔxout     = max(maxcΔxout, maxcΔx)
        
        prob.etime += prob.Δt

        if timestepping
            prob.Δt = Δt_stepping(prob.buf, prob.∑∂v∂t, prob.v, x, prob.c₀, prob.h, prob.CFL, timelims)
        end

    end
    # update summs and gradiends after bath 
    fill!(prob.∑W, zero(T))
    fill!(prob.∑∇W[1], zero(T))
    fill!(prob.∑∇W[2], zero(T))
    ∑W_2d!(prob.∑W, pairs, x, prob.sphkernel, prob.H⁻¹)
    ∑∇W_2d!(prob.∑∇W, pairs, x, prob.sphkernel, prob.H⁻¹)
    updaten, maxcΔxout
end
