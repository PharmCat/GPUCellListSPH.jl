
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

mutable struct SPHProblem
    system::GPUCellList
    dim::Int
    h::Float64
    h⁻¹::Float64
    H::Float64
    H⁻¹::Float64
    sphkernel
    ∑W
    ∑∇W
    ∇Wₙ
    ∑∂Π∂t
    ∑∂v∂t
    ∑∂ρ∂t
    ρ
    ρΔt½
    v
    vΔt½
    xΔt½
    ml
    gf
    isboundary
    ρ₀::Float64
    m₀::Float64
    Δt::Float64
    α::Float64
    g::Float64
    c₀::Float64
    γ
    s::Float64                # surface tension
    δᵩ::Float64
    CFL::Float64
    buf
    etime::Float64
    function SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64; s::Float64 = 0.0)

        dim = length(CUDA.@allowscalar first(system.points))
        N   = length(system.points)

        ∑W      = CUDA.zeros(Float64, N)
        ∑∇W     = CUDA.zeros(Float64, N, dim)
        ∇Wₙ     = CUDA.fill(zero(NTuple{dim, Float64}), length(system.pairs))
        ∑∂ρ∂t   = CUDA.zeros(Float64, N)
        ∑∂Π∂t   = CUDA.zeros(Float64, N, dim)
        ∑∂v∂t   = CUDA.zeros(Float64, N, dim)

        buf     = CUDA.zeros(Float64, N)

        ρΔt½    = CUDA.deepcopy(ρ)
        vΔt½    = CUDA.deepcopy(v)
        xΔt½    = CUDA.deepcopy(system.points)

        new{}(system, dim, h, 1/h, H, 1/H, sphkernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t, ρ, ρΔt½, v, vΔt½, xΔt½, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, s, δᵩ, CFL, buf, 0.0)
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
function _stepsolve!(prob::SPHProblem, n::Int, ::StepByStep; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))
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
        # kernels sum for each cell
        ∑W_2d!(prob.∑W, pairs, prob.sphkernel, prob.H⁻¹)
        # kernels gradient  for each cell (∑∇W) and value for each pair (∇Wₙ)
        ∑∇W_2d!(prob.∑∇W, prob.∇Wₙ, pairs, x, prob.sphkernel, prob.H⁻¹)
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
        update_ρ!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.isboundary)
        # calc v at Δt½
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ml) 
        # calc x at Δt½
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5, prob.ml)

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


function get_points(prob::SPHProblem)
    prob.system.points
end

function get_velocity(prob::SPHProblem)
    prob.v
end

function get_density(prob::SPHProblem)
    prob.ρ
end

function get_acceleration(prob::SPHProblem)
    prob.∑∂v∂t
end

function get_simtime(prob::SPHProblem)
    prob.etime
end

function get_dt(prob::SPHProblem)
    prob.Δt
end

"""
    timesolve!(prob::SPHProblem; batch = 10, timeframe = 1.0, writetime = 0, path = nothing, pvc = false, timestepping = false, timelims = (-Inf, +Inf), anim = false) 

Make simulation by `batch` iterations within `timeframe`. 

writetime - time interval for write vtk / animation.

path - path to export directory.

anim - make animation.

showframe - show animation each frame.
"""
function timesolve!(prob::SPHProblem; batch = 10, timeframe = 1.0, writetime = 0, path = nothing, pvc::Bool = false, timestepping = false, timelims = (-Inf, +Inf), anim::Bool = false, showframe::Bool = true) 

    nt = prob.etime + writetime
    i  = 0
    if writetime > 0 && !isnothing(path) 
        if pvc
            pvd = paraview_collection(joinpath(path, "OUTPUT_PVC"))
            add_timestep(joinpath(path, "OUTPUT_"*lpad(i, 5, "0")), pvd, prob.etime, get_points(prob), get_density(prob), get_acceleration(prob), get_velocity(prob))
        else
            create_vtp_file(joinpath(path, "OUTPUT_"*lpad(i, 5, "0")), get_points(prob), get_density(prob), get_acceleration(prob), get_velocity(prob))
        end
    end
    prog = ProgressUnknown(desc = "Calculating...:", spinner=true, showspeed=true)

    if anim
        animation = Animation()
    end    

    while prob.etime <= timeframe
       
        stepsolve!(prob, batch; timestepping = timestepping, timelims = timelims)

        if writetime > 0  && nt < prob.etime
            nt += writetime
            cpupoints = Array(get_points(prob))
            if !isnothing(path)
                if pvc
                    add_timestep(joinpath(path, "OUTPUT_"*lpad(i, 5, "0")), pvd, prob.etime, cpupoints, get_density(prob), get_acceleration(prob), get_velocity(prob))
                else
                    create_vtp_file(joinpath(path, "OUTPUT_"*lpad(i,5,"0")), get_points(prob), cpupoints, get_acceleration(prob), get_velocity(prob))
                end
            end
            if anim
                ax = map(x->x[1], cpupoints)
                ay = map(x->x[2], cpupoints) 
                p = scatter(ax, ay, leg = false)
                if showframe display(p) end
                frame(animation, p)
            end
        end


        i += 1
        next!(prog, spinner="🌑🌒🌓🌔🌕🌖🌗🌘", showvalues = [(:iter, i), (:time, prob.etime), (:Δt, prob.Δt)])
    end

    if writetime > 0 && !isnothing(path) 
        if pvc
            vtk_save(pvd)
        end
        if anim
            gif(animation, joinpath(path, "anim_output_fps30.gif"), fps = Int(ceil(1/writetime)))
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
end


