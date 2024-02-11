
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
    δᵩ::Float64
    CFL::Float64
    buf
    etime::Float64
    function SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64)

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

        new{}(system, dim, h, 1/h, H, 1/H, sphkernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t, ρ, ρΔt½, v, vΔt½, xΔt½, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL, buf, 0.0)
    end
end

"""
    stepsolve!(prob::SPHProblem, n::Int = 1; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))

Make n itarations. 

timestepping - call Δt_stepping for adjust Δt

timelims - minimal and maximum values for Δt
"""
function stepsolve!(prob::SPHProblem, n::Int = 1; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))
    if timestepping || timelims[1] > timelims[1] error("timelims[1] should be > timelims[2]") end
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

        ∑W_2d!(prob.∑W, pairs, prob.sphkernel, prob.H⁻¹)
        #if isnan(minimum(prob.∑W)) error("1") end 
        ∑∇W_2d!(prob.∑∇W, prob.∇Wₙ, pairs, x, prob.sphkernel, prob.H⁻¹)
        #if isnan(minimum(x->x[1], prob.∑∇W)) error("2") end 
        ∂ρ∂tDDT!(prob.∑∂ρ∂t, prob.∇Wₙ, pairs, x, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 
        #if isnan(minimum(prob.∑∂ρ∂t)) error("3") end 
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        #if isnan(minimum(prob.∑∂Π∂t)) error("4") end 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρ, prob.c₀, prob.γ, prob.ρ₀) 
        #if isnan(minimum(prob.∑∂v∂t)) error("5") end 
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t,  gravvec(prob.g, prob.dim), prob.gf) 
        #if isnan(minimum(prob.∑∂v∂t)) error("6") end 
        update_ρ!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.isboundary)
        #if isnan(minimum(prob.ρΔt½)) error("7") end 
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ml) 
        #if isnan(minimum(x->x[1], prob.vΔt½)) error("8") end 
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5, prob.ml)
        #if isnan(minimum(x->x[1], prob.xΔt½)) error("9") end 
        fill!(prob.∑∂ρ∂t, zero(Float64))
        fill!(prob.∑∂Π∂t, zero(Float64))
        fill!(prob.∑∂v∂t, zero(Float64))

        ∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 
        #if isnan(minimum(prob.∑∂ρ∂t)) error("10") end 
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        #if isnan(minimum(prob.∑∂Π∂t)) error("11") end 
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρ, prob.c₀, prob.γ, prob.ρ₀) 
        #if isnan(minimum(prob.∑∂v∂t)) error("12") end 
        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t, gravvec(prob.g, prob.dim), prob.gf)
        #if isnan(minimum(prob.∑∂v∂t)) error("13") end 

        update_all!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.ρ₀, prob.isboundary, prob.ml)
        #if isnan(minimum(prob.ρ)) error("14") end 
        #if isnan(minimum(x->x[1], x)) error("15") end 
        #if isnan(minimum(x->x[1], prob.v)) error("16") end 

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
    timesolve!(prob::SPHProblem; batch = 10, timeframe = 1.0, vtkwritetime = 0, vtkpath = nothing) 

Make simulation by `batch` iterations within `timeframe`. 

vtkwritetime - time interval for write vtk.

vtkpath - path to vtk directory.
"""
function timesolve!(prob::SPHProblem; batch = 10, timeframe = 1.0, vtkwritetime = 0, vtkpath = nothing, timestepping = false, timelims = (-Inf, +Inf)) 

    nt = prob.etime + vtkwritetime
    i  = 0
    if vtkwritetime > 0 && !isnothing(vtkpath) 
        create_vtp_file(joinpath(vtkpath, "OUTPUT_"*lpad(i, 5, "0")), get_points(prob), get_density(prob), get_acceleration(prob), get_velocity(prob))
    end
    prog = ProgressUnknown(desc = "Calculating...:", spinner=true, showspeed=true)

    while prob.etime <= timeframe
       
        stepsolve!(prob, batch; timestepping = timestepping, timelims = timelims)

        if vtkwritetime > 0 && !isnothing(vtkpath) && nt < prob.etime
            nt += vtkwritetime
            create_vtp_file(joinpath(vtkpath, "OUTPUT_"*lpad(i,5,"0")), get_points(prob), get_density(prob), get_acceleration(prob), get_velocity(prob))
        end

        i += 1
        next!(prog, spinner="🌑🌒🌓🌔🌕🌖🌗🌘", showvalues = [(:iter, i), (:time, prob.etime), (:Δt, prob.Δt)])
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


