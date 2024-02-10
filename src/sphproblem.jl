
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

        new{}(system, dim, h, 1/h, H, 1/H, sphkernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t, ρ, ρΔt½, v, vΔt½, xΔt½, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL, buf)
    end
end


function stepsolve!(prob::SPHProblem, n::Int = 1)
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

        ∑∇W_2d!(prob.∑∇W, prob.∇Wₙ, pairs, x, prob.sphkernel, prob.H⁻¹)

        ∂ρ∂tDDT!(prob.∑∂ρ∂t, prob.∇Wₙ, pairs, x, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 

        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, x, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
    
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρ, prob.c₀, prob.γ, prob.ρ₀) 

        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t,  gravvec(prob.g, prob.dim), prob.gf) 

        update_ρ!(prob.ρΔt½, prob.∑∂ρ∂t, prob.Δt * 0.5, prob.ρ₀, prob.isboundary)
    
        update_vp∂v∂tΔt!(prob.vΔt½, prob.∑∂v∂t, prob.Δt * 0.5, prob.ml) 
 
        update_xpvΔt!(prob.xΔt½, prob.vΔt½, prob.Δt * 0.5, prob.ml)

        fill!(prob.∑∂ρ∂t, zero(Float64))
        fill!(prob.∑∂Π∂t, zero(Float64))
        fill!(prob.∑∂v∂t, zero(Float64))

        ∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 
        ∂Π∂t!(prob.∑∂Π∂t, prob.∇Wₙ, pairs, prob.xΔt½, prob.h, prob.ρ, prob.α, prob.v, prob.c₀, prob.m₀)
        ∂v∂t!(prob.∑∂v∂t,  prob.∇Wₙ, pairs,  prob.m₀, prob.ρ, prob.c₀, prob.γ, prob.ρ₀) 

        completed_∂v∂t!(prob.∑∂v∂t, prob.∑∂Π∂t, gravvec(prob.g, prob.dim), prob.gf)

        update_all!(prob.ρ, prob.ρΔt½, prob.v, prob.vΔt½, x, prob.xΔt½, prob.∑∂ρ∂t, prob.∑∂v∂t, prob.Δt, prob.ρ₀, prob.isboundary, prob.ml)
    
        prob.Δt = Δt_stepping(prob.buf, prob.∑∂v∂t, prob.v, x, prob.c₀, prob.h, prob.CFL)
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


