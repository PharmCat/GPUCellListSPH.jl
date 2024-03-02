
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

system::GPUCellList{T} - system of particles (position and cells);

dx - dx;

h - smoothing length;

H- kernel support radius (2h);

sphkernel::AbstractSPHKernel - SPH kernel from SPHKernels.jl;

ρ - rho (vector);

v - velocity (vector);

ptype - particle type: 1 - fluid 1; 0 - boundary; -1 boundary hard layer;

ρ₀ - Reference density;

m₀ - nitial mass;

Δt - default Δt;

α - Artificial viscosity alpha constant;

g - gravity constant;

c₀ - speed of sound;

γ - Gamma, 7 for water (used in the pressure equation of state);

δᵩ- Coefficient for density diffusion, typically 0.1;

CFL - CFL number for the simulation.
"""
mutable struct SPHProblem{T}
    system::GPUCellList
    dim::Int
    dx::T
    h::T                                        # smoothing length
    h⁻¹::T
    H::T                                        # kernel support radius (2h)
    H⁻¹::T
    sphkernel::AbstractSPHKernel                # SPH kernel from SPHKernels.jl
    ∑W::CuArray                                 # sum of kernel values
    ∑∇W                                         # sum of kernel gradients
    W::CuArray                                  # values of kernel gradient for each pair 
    ∇W::CuArray                                 # values of kernel gradient for each pair 
    ∑∂v∂t                                       # acceleration (momentum equation)
    ∑∂ρ∂t                                       # rho diffusion - density derivative function (with diffusion)
    ρ::CuArray                                  # rho
    ρΔt½::CuArray                               # rho at t½  
    v::CuArray                                  # velocity
    vΔt½::CuArray                               # velocity at t½  
    xΔt½::CuArray                               # coordinates at xΔt½
    P::CuArray                                  # pressure (Equation of State in Weakly-Compressible SPH)
    ptype::CuArray                              # particle type: 1 - fluid 1; 0 - boundary; -1 boundary hard layer 
    ρ₀::T                                 # Reference density
    m₀::T                                 # Initial mass
    Δt::T                                 # default Δt
    α::T                                  # Artificial viscosity alpha constant
    𝜈::T                                  # kinematic fluid viscosity
    g::T                                  # gravity constant
    c₀::T                                 # speed of sound
    γ                                           # Gamma, 7 for water (used in the pressure equation of state)
    s::T                                  # surface tension constant
    δᵩ::T                                 # Coefficient for density diffusion, typically 0.1
    CFL::T                                # CFL number for the simulation 
    buf::CuArray                                # buffer for dt calculation
    buf2                                # buffer 
    etime::T                              # simulation time
    cΔx                                         # cumulative location changes in batch
    nui::T                                # non update interval, update if maximum(maximum.(abs, prob.cΔx)) > 0.9 * prob.nui  
    # Dynamic Particle Collision (DPC) 
    dpc_l₀::T       # minimal distance
    dpc_pmin::T     # minimal pressure
    dpc_pmax::T     # maximum pressure
    dpc_λ::T        # λ is a non-dimensional adjusting parameter
    # XSPH
    xsph_𝜀::T       # xsph constant
    function SPHProblem(system::GPUCellList{T}, dx, h::Float64, H::Float64, sphkernel::AbstractSPHKernel, ρ, v, ptype, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64; s::Float64 = 0.0) where T <: AbstractFloat

        dim = length(CUDA.@allowscalar first(system.points))
        N   = length(system.points)

        ∑W      = CUDA.zeros(T, N)
        ∑∇W     = Tuple(CUDA.zeros(T, N) for n in 1:dim)
        W       = CUDA.zeros(T, length(system.pairs))
        ∇W      = CUDA.fill(zero(NTuple{dim, T}), length(system.pairs))
        ∑∂ρ∂t   = CUDA.zeros(T, N)

        ∑∂v∂t   = Tuple(CUDA.zeros(T, N) for n in 1:dim)

        buf     = CUDA.zeros(T, N)

        buf2    = Tuple(CUDA.zeros(T, N) for n in 1:dim)

        ρΔt½    = CUDA.deepcopy(ρ)
        vΔt½    = CUDA.deepcopy(v)
        xΔt½    = CUDA.deepcopy(system.points)
        cΔx     = Tuple(CUDA.zeros(T, N) for n in 1:dim)
        P       = CUDA.zeros(T, N)
        new{T}(system, 
        dim, 
        dx,
        h, 
        1/h, 
        H, 
        1/H, 
        sphkernel, 
        ∑W, 
        ∑∇W, 
        W, 
        ∇W, 
        ∑∂v∂t, 
        ∑∂ρ∂t, 
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
        0.0,
        g, 
        c₀, 
        γ, 
        s, 
        δᵩ, 
        CFL, 
        buf,
        buf2,
        0.0, 
        cΔx, 
        system.dist - H, 
        0.0, 
        1.0, 
        10000.0, 
        0.01,
        0.0)
    end
end

"""
    stepsolve!(prob::SPHProblem, n::Int = 1; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))

Make `n` itarations. 

timestepping - call Δt_stepping for adjust Δt

timelims - minimal and maximum values for Δt
"""
function stepsolve!(prob::SPHProblem, n::Int = 1; simwl::SimWorkLoad = StepByStep(), kwargs...)
    _stepsolve!(prob, n, simwl;  kwargs...)
end
function _stepsolve!(prob::SPHProblem{T}, n::Int, ::StepByStep; timestepping = false, timelims = (sqrt(eps()), prob.CFL * prob.H /3prob.c₀), verbode = true) where T
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


