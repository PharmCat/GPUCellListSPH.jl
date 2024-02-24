
function GPUCellListSPH.zero(::Type{NTuple{2, T}}) where T
    (zero(T), zero(T))
end

function GPUCellListSPH.zero(::Type{NTuple{3, T}}) where T
    (zero(T), zero(T), zero(T))
end

function gravvec(g::T, dim::Int) where T
    if dim == 2
        return (zero(T), g)
    elseif dim == 3
        return (zero(T), g, zero(T))
    end
    (g,)
end

abstract type SimWorkLoad end
struct StepByStep <: SimWorkLoad end
struct Effective  <: SimWorkLoad end

"""
    SPHProblem(system::GPUCellList, h::Float64, H::Float64, sphkernel::AbstractSPHKernel, ρ, v, ptype, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64; s::Float64 = 0.0)

SPH simulation data structure.
"""
mutable struct SPHProblem
    system::GPUCellList
    dim::Int
    h::Float64                                  # smoothing length
    h⁻¹::Float64
    H::Float64                                  # kernel support radius (2h)
    H⁻¹::Float64
    sphkernel::AbstractSPHKernel                # SPH kernel from SPHKernels.jl
    ∑W::CuArray                                 # sum of kernel values
    ∑∇W                                         # sum of kernel gradients
    W::CuArray                                  # values of kernel gradient for each pair 
    ∇W::CuArray                                 # values of kernel gradient for each pair 
    ∑∂Π∂t                                       # artificial viscosity
    ∑∂v∂t                                       # acceleration (momentum equation)
    ∑∂ρ∂t                                       # rho diffusion - density derivative function (with diffusion)
    ∑Δvdpc                                      # velocity dynamic particle collision correction 
    ρ::CuArray                                  # rho
    ρΔt½::CuArray                               # rho at t½  
    v::CuArray                                  # velocity
    vΔt½::CuArray                               # velocity at t½  
    xΔt½::CuArray                               # coordinates at xΔt½
    P::CuArray                                  # pressure (Equation of State in Weakly-Compressible SPH)
    ptype::CuArray                              # particle type: 1 - fluid 1; 0 - boundary; -1 boundary hard layer 
    ρ₀::Float64                                 # Reference density
    m₀::Float64                                 # Initial mass
    Δt::Float64                                 # default Δt
    α::Float64                                  # Artificial viscosity alpha constant
    g::Float64                                  # gravity constant
    c₀::Float64                                 # speed of sound
    γ                                           # Gamma, 7 for water (used in the pressure equation of state)
    s::Float64                                  # surface tension constant
    δᵩ::Float64                                 # Coefficient for density diffusion, typically 0.1
    CFL::Float64                                # CFL number for the simulation 
    buf::CuArray                                # buffer for dt calculation
    etime::Float64                              # simulation time
    cΔx                                         # cumulative location changes in batch
    nui::Float64                                # non update interval, update if maximum(maximum.(abs, prob.cΔx)) > 0.9 * prob.nui  
    # Dynamic Particle Collision (DPC) 
    dpc_l₀::Float64       # minimal distance
    dpc_pmin::Float64     # minimal pressure
    dpc_pmax::Float64     # maximum pressure
    dpc_λ::Float64        # λ is a non-dimensional adjusting parameter
    function SPHProblem(system::GPUCellList, h::Float64, H::Float64, sphkernel::AbstractSPHKernel, ρ, v, ptype, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64; s::Float64 = 0.0)

        dim = length(CUDA.@allowscalar first(system.points))
        N   = length(system.points)

        ∑W      = CUDA.zeros(Float64, N)
        ∑∇W     = Tuple(CUDA.zeros(Float64, N) for n in 1:dim)
        W       = CUDA.zeros(Float64, length(system.pairs))
        ∇W      = CUDA.fill(zero(NTuple{dim, Float64}), length(system.pairs))
        ∑∂ρ∂t   = CUDA.zeros(Float64, N)

        ∑∂Π∂t   = Tuple(CUDA.zeros(Float64, N) for n in 1:dim)

        ∑∂v∂t   = Tuple(CUDA.zeros(Float64, N) for n in 1:dim)

        ∑Δvdpc = Tuple(CUDA.zeros(Float64, N) for n in 1:dim)

        buf     = CUDA.zeros(Float64, N)

        ρΔt½    = CUDA.deepcopy(ρ)
        vΔt½    = CUDA.deepcopy(v)
        xΔt½    = CUDA.deepcopy(system.points)
        cΔx     = Tuple(CUDA.zeros(Float64, N) for n in 1:dim)
        P       = CUDA.zeros(Float64, N)
        new{}(system, 
        dim, 
        h, 
        1/h, 
        H, 
        1/H, 
        sphkernel, 
        ∑W, 
        ∑∇W, 
        W, 
        ∇W, 
        ∑∂Π∂t, 
        ∑∂v∂t, 
        ∑∂ρ∂t, 
        ∑Δvdpc, 
        ρ, 
        ρΔt½, 
        v, 
        vΔt½, 
        xΔt½, 
        P, 
        ptype, 
        ρ₀, 
        m₀, 
        Δt, 
        α, 
        g, 
        c₀, 
        γ, 
        s, 
        δᵩ, 
        CFL, 
        buf, 
        0.0, 
        cΔx, 
        system.dist - H, 
        0.0, 
        1.0, 
        10000.0, 
        0.01)
    end
end

"""
    stepsolve!(prob::SPHProblem, n::Int = 1; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))

Make n itarations. 

timestepping - call Δt_stepping for adjust Δt

timelims - minimal and maximum values for Δt
"""
function stepsolve!(prob::SPHProblem, n::Int = 1; simwl::SimWorkLoad = StepByStep(), kwargs...)
    _stepsolve!(prob, n, simwl;  kwargs...)
end
function _stepsolve!(prob::SPHProblem, n::Int, ::StepByStep; timestepping = false, timelims = (sqrt(eps()), prob.CFL * prob.H /3prob.c₀), verbode = true)
    if timestepping && timelims[1] > timelims[1] error("timelims[1] should be < timelims[2]") end

    x              = prob.system.points
    pairs          = neighborlist(prob.system)
    skipupdate     = false
    updaten        = 0
    skipupdaten    = 0
    maxcΔx         = 0.0
    maxcΔxout      = 0.0

    dpckernlim = find_zero(x-> 1.0 - 𝒲(prob.sphkernel, x, 1.0), 0.5)

    for iter = 1:n
        if skipupdate 
            skipupdaten += 1
        else
            update!(prob.system)
            x           = prob.system.points
            pairs       = neighborlist(prob.system)
            sort!(pairs, by = first)
            for a in prob.cΔx fill!(a, zero(Float64)) end
            skipupdate  = true
            updaten += 1 
        end


        fill!(prob.∑∂ρ∂t, zero(Float64))

        fill!(prob.∑∂Π∂t[1], zero(Float64))
        fill!(prob.∑∂v∂t[1], zero(Float64))
        #fill!(prob.∑∂v∂tdpc[1], zero(Float64))

        fill!(prob.∑∂Π∂t[2], zero(Float64))
        fill!(prob.∑∂v∂t[2], zero(Float64))
        #fill!(prob.∑∂v∂tdpc[2], zero(Float64))


        if length(prob.∇W) != length(pairs)
            CUDA.unsafe_free!(prob.∇W)
            CUDA.unsafe_free!(prob.W)
            prob.∇W =  CUDA.fill((zero(Float64), zero(Float64)), length(pairs)) # DIM = 2
            prob.W =  CUDA.fill(zero(Float64), length(pairs))
        end
        # kernels for each pair
        W_2d!(prob.W, pairs, x, prob.sphkernel, prob.H⁻¹)
        # kernels gradientfor each pair
        ∇W_2d!(prob.∇W, pairs, x, prob.sphkernel, prob.H⁻¹)
        # density derivative with density diffusion
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, prob.∇W, pairs, x, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ptype) 
        # artificial viscosity
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇W, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        #  pressure
        pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇W, prob.P, pairs,  prob.m₀, prob.ρ) 
        # add gravity and artificial viscosity 
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t,  gravvec(prob.g, prob.dim)) 
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, x, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        
        # following steps (update_ρ!, update_vp∂v∂tΔt!, update_xpvΔt!) can be done in one kernel 
        # calc ρ at Δt½
        update_ρ!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.ptype)
        # calc v at Δt½
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ptype) 
        # calc x at Δt½
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5)

        # set derivative to zero for Δt½ calc

        fill!(prob.∑∂ρ∂t, zero(Float64))

        fill!(prob.∑∂Π∂t[1], zero(Float64))
        fill!(prob.∑∂v∂t[1], zero(Float64))
        fill!(prob.∑Δvdpc[1], zero(Float64))

        fill!(prob.∑∂Π∂t[2], zero(Float64))
        fill!(prob.∑∂v∂t[2], zero(Float64))
        fill!(prob.∑Δvdpc[2], zero(Float64))

        # density derivative with density diffusion at  xΔt½ 
        ∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇W, pairs, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ptype) 
        # artificial viscosity at xΔt½ 
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇W, pairs, prob.xΔt½, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        #  pressure
        pressure!(prob.P, prob.ρ, prob.c₀, prob.γ, prob.ρ₀, prob.ptype) 
        # momentum equation 
        ∂v∂t!(prob.∑∂v∂t, prob.∇W, prob.P, pairs,  prob.m₀, prob.ρ)
        # add gravity and artificial viscosity
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t, gravvec(prob.g, prob.dim))
        # add surface tension if s > 0
        if prob.s > 0
            ∂v∂tpF!(prob.∑∂v∂t, pairs, prob.xΔt½, prob.s, prob.h, prob.m₀, prob.ptype) 
        end
        # update all with symplectic position Verlet scheme
        update_all!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.cΔx, prob.ρ₀, prob.ptype)
        
        # Dynamic Particle Collision (DPC) 
        if prob.dpc_l₀ > 0 && prob.dpc_λ > 0
            dpcreg!(prob.∑Δvdpc, prob.v, prob.ρ, prob.P, pairs, x, prob.sphkernel, prob.dpc_l₀, prob.dpc_pmin, prob.dpc_pmax, prob.Δt, prob.dpc_λ, dpckernlim)  
            update_dpcreg!(prob.v, x, prob.∑Δvdpc, prob.Δt, prob.ptype)
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
    fill!(prob.∑W, zero(Float64))
    fill!(prob.∑∇W[1], zero(Float64))
    fill!(prob.∑∇W[2], zero(Float64))
    ∑W_2d!(prob.∑W, pairs, x, prob.sphkernel, prob.H⁻¹)
    ∑∇W_2d!(prob.∑∇W, pairs, x, prob.sphkernel, prob.H⁻¹)
    updaten, maxcΔxout
end


function get_points(prob::SPHProblem)
    prob.system.points
end

function get_velocity(prob::SPHProblem)
    prob.v
end

function get_density(prob::SPHProblem)
    prob.ρ
end

function get_pressure(prob::SPHProblem)
    prob.P
end

function get_acceleration(prob::SPHProblem)
    prob.∑∂v∂t
end

function get_dpccorr(prob::SPHProblem)
    prob.∑Δvdpc
end

function get_simtime(prob::SPHProblem)
    prob.etime
end

function get_dt(prob::SPHProblem)
    prob.Δt
end

function get_sumw(prob::SPHProblem)
    prob.∑W
end

function get_sumgradw(prob::SPHProblem)
    prob.∑∇W
end

"""
    timesolve!(prob::SPHProblem; batch = 10, 
    timeframe = 1.0, 
    writetime = 0, 
    path = nothing, 
    pvc = false, 
    vtkvars = ["Acceleration", "Velocity", "Pressure"],
    timestepping = false, 
    timelims = (-Inf, +Inf), 
    anim = false,
    plotsettings = Dict(:leg => false)) 

Make simulation by `batch` iterations within `timeframe`. 

writetime - time interval for write vtk / animation.

path - path to export directory.

pvc - make PVD file.

vtkvars - variables for export, full list:  `["Acceleration", "Velocity", "Pressure", "Density", "∑W", "∑∇W", "DPC"]` 

anim - make animation.

showframe - show animation each frame.

plotsettings - keywords for plotting.
"""
function timesolve!(prob::SPHProblem; batch = 10, timeframe = 1.0, 
    writetime = 0, 
    path = nothing, 
    pvc::Bool = false, 
    vtkvars = ["Acceleration", "Velocity", "Pressure"],
    timestepping = false, 
    timelims = (sqrt(eps()), prob.CFL * prob.H /3prob.c₀), 
    anim::Bool = false, 
    showframe::Bool = true, 
    verbose = true, 
    plotsettings = Dict(:leg => false)) 

    if timelims[2] > prob.CFL * prob.H /3prob.c₀ 
        @warn "Maximum dt limit ($(timelims[2])) > CFL*H/3c₀ ($(prob.CFL * prob.H /3prob.c₀))" 
    end
    if timestepping timelims = (max(timelims[1], eps()), min(timelims[2], prob.CFL * prob.H /3prob.c₀, prob.Δt)) end
    if verbose
        println("    Start simulation...")
        println("Timestepping: $timestepping")
        if timestepping println("Δt limitss: $timelims") end
        println("Batch: $batch")
        println("NUI: $(prob.nui)")
        
    end
    nt = prob.etime + writetime
    i  = 0
    if writetime > 0 && !isnothing(path)
        expdict                 = Dict()
        cpupoints               = Array(get_points(prob))
        coordsarr               = [map(x -> x[i], cpupoints) for i in 1:length(first(cpupoints))]
        if "Density"      in vtkvars expdict["Density"]      = Array(get_density(prob)) end
        if "Pressure"     in vtkvars expdict["Pressure"]     = Array(get_pressure(prob)) end
        if "Acceleration" in vtkvars expdict["Acceleration"] = Array.(get_acceleration(prob)) end
        if "Velocity" in vtkvars 
            av                      = Array(get_velocity(prob))
            expdict["Velocity"]     = permutedims(hcat([map(x -> x[i], av) for i in 1:length(first(av))]...))
        end
        if "∑W" in vtkvars expdict["∑W"]           = Array(get_sumw(prob)) end
        if "∑∇W" in vtkvars expdict["∑∇W"]         = Array.(get_sumgradw(prob)) end
        if "DPC" in vtkvars expdict["DPC"]         = Array.(get_dpccorr(prob)) end
       
        if pvc
            pvd = paraview_collection(joinpath(path, "OUTPUT_PVC"))
        else
            pvd = nothing
        end
        create_vtp_file(joinpath(path, "OUTPUT_"*lpad(i, 5, "0")), coordsarr, expdict, pvd, prob.etime)
    end
    prog = ProgressUnknown(desc = "Calculating...:", spinner=true, showspeed=true)

    if anim
        animation = Animation()
    end    

    local diaginf
    
    while prob.etime <= timeframe
       
        diaginf = stepsolve!(prob, batch; timestepping = timestepping, timelims = timelims)

        if writetime > 0  && nt < prob.etime
            nt += writetime

            if !isnothing(path)
                expdict                 = Dict()
                cpupoints               = Array(get_points(prob))
                coordsarr               = [map(x -> x[i], cpupoints) for i in 1:length(first(cpupoints))]
                if "Density"      in vtkvars expdict["Density"]      = Array(get_density(prob)) end
                if "Pressure"     in vtkvars expdict["Pressure"]     = Array(get_pressure(prob)) end
                if "Acceleration" in vtkvars expdict["Acceleration"] = Array.(get_acceleration(prob)) end
                 if "Velocity" in vtkvars 
                    av                      = Array(get_velocity(prob))
                    expdict["Velocity"]     = permutedims(hcat([map(x -> x[i], av) for i in 1:length(first(av))]...))
                end
                if "∑W" in vtkvars expdict["∑W"]           = Array(get_sumw(prob)) end
                if "∑∇W" in vtkvars expdict["∑∇W"]         = Array.(get_sumgradw(prob)) end
                if "DPC" in vtkvars expdict["DPC"]         = Array.(get_dpccorr(prob)) end

                create_vtp_file(joinpath(path, "OUTPUT_"*lpad(i, 5, "0")), coordsarr, expdict, pvd, prob.etime)
            end
            if anim
                ax = map(x->x[1], cpupoints)
                ay = map(x->x[2], cpupoints) 
                p = scatter(ax, ay; plotsettings...)
                if showframe display(p) end
                frame(animation, p)
            end
        end

        i += 1
        next!(prog, spinner="🌑🌒🌓🌔🌕🌖🌗🌘", showvalues = [(:iter, i), (:time, prob.etime), (:Δt, prob.Δt), (Symbol("updn"), diaginf[1]), (:dxpncu, diaginf[2])])
    end

    if writetime > 0 && !isnothing(path) 
        if pvc
            vtk_save(pvd)
        end
        if anim
            fps = Int(ceil(1/writetime))
            gif(animation, joinpath(path, "anim_output_fps$(fps).gif"), fps = fps)
        end
    end
    finish!(prog)
end


function Base.show(io::IO, p::SPHProblem)
    println(io, "  SPH Problem ")
    println(io, p.system)
    println(io, "  h: ", p.h)
    println(io, "  H: ", p.H)
    println(io, "  SPH Kernel: ", p.sphkernel)
    println(io, "  E Time: ", p.etime)
    if p.s > 0 
        println(io, "  Surface tension: ", p.s)
    else
        println(io, "  Surface tension: not used")
    end
    if p.dpc_l₀ > 0 && p.dpc_λ > 0 
        println(io, "  DPC: l₀ = ", p.dpc_l₀, " , λ = ", p.dpc_λ)
    else
        println(io, "  DPC: not used")
    end
end


