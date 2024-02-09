using BenchmarkTools, GPUCellListSPH, CUDA

cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 2)))

system = GPUCellListSPH.GPUCellList(cpupoints, (0.016, 0.016), 0.016)

system.points # points

system.pairs # pairs list

system.grid # cell grid 

sum(system.cellpnum) # total cell number

maximum(system.cellpnum) # maximum particle in cell

count(x-> !isnan(x[3]), system.pairs)  == system.pairsn


GPUCellListSPH.update!(system)

GPUCellListSPH.partialupdate!(system)

count(x-> !isnan(x[3]), system.pairs) == system.pairsn

@benchmark GPUCellListSPH.update!($system)


@benchmark GPUCellListSPH.partialupdate!($system)

using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "./input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "./input/BoundaryPoints_Dp0.02.csv")


    
    cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv)

    ρ   = cu(Array([DF_FLUID.Rhop;DF_BOUND.Rhop]))
    ρΔt½  = copy(ρ)
    ml  = cu([ ones(size(DF_FLUID,1)) ; zeros(size(DF_BOUND,1))])

    isboundary  = .!Bool.(ml)

    gf = cu([-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))])
    v   = CUDA.fill((0.0, 0.0), length(cpupoints))
    vΔt½  = copy(v)

    a   = CUDA.zeros(Float64, length(cpupoints))

    dx  = 0.02
    h   = 1.2 * sqrt(2) * dx
    H   = 2h
    h⁻¹ = 1/h
    H⁻¹ = 1/H
    dist = H
    ρ₀  = 1000
    m₀  = ρ₀ * dx * dx #mᵢ  = mⱼ = m₀
    α   = 0.01
    g   = 9.81
    c₀  = sqrt(g * 2) * 20
    γ   = 7
    Δt  = dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.2

    cellsize = (H, H)
    x = gpupoints = cu(cpupoints)
    xΔt½ = copy(gpupoints)

    N      = length(cpupoints)

    sphkernel    = WendlandC2(Float64, 2)

    system  = GPUCellListSPH.GPUCellList(cpupoints, cellsize, H)
 
    sumW    = CUDA.zeros(Float64, N)
    sum∇W   = CUDA.zeros(Float64, N, 2)
    ∇Wₙ     =  CUDA.fill((zero(Float64), zero(Float64)), length(system.pairs))
    ∑∂ρ∂t   = CUDA.zeros(Float64, N)
    ∑∂Π∂t   = CUDA.zeros(Float64, N, 2)
    ∑∂v∂t   = CUDA.zeros(Float64, N, 2)

#== ==#
function sph_simulation(system, sphkernel, ρ, ρΔt½, v, vΔt½, xΔt½, ∑∂Π∂t, ∑∂ρ∂t, ∑∂v∂t, sumW, sum∇W, ∇Wₙ, Δt, ρ₀, isboundary, ml, h, H⁻¹, m₀, δᵩ, c₀, γ, g, α)

    GPUCellListSPH.update!(system)

    x     = system.points
    pairs = system.pairs

    fill!(sumW, zero(Float64))
    fill!(∑∂ρ∂t, zero(Float64))
    fill!(∑∂Π∂t, zero(Float64))
    fill!(∑∂v∂t, zero(Float64))

    if length(∇Wₙ) != length(pairs)
        ∇Wₙ =  CUDA.fill((zero(Float64), zero(Float64)), length(system.pairs))
    end

    GPUCellListSPH.∑W_2d!(sumW, pairs, sphkernel, H⁻¹)

    GPUCellListSPH.∑∇W_2d!(sum∇W, ∇Wₙ, pairs, x, sphkernel, H⁻¹)

    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t, ∇Wₙ, pairs, x, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml) 

    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, x, h, ρ, α, v, c₀, m₀)
    
    GPUCellListSPH.∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs,  m₀, ρ, c₀, γ, ρ₀) 

    GPUCellListSPH.completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  (0.0, g), gf)

    GPUCellListSPH.update_ρ!(ρΔt½, ∑∂ρ∂t, Δt/2, ρ₀, isboundary)
    
    GPUCellListSPH.update_vp∂v∂tΔt!(vΔt½, ∑∂v∂t, Δt/2, ml) 

    GPUCellListSPH.update_xpvΔt!(xΔt½, vΔt½, Δt/2, ml)

    fill!(∑∂ρ∂t, zero(Float64))
    fill!(∑∂Π∂t, zero(Float64))
    fill!(∑∂v∂t, zero(Float64))

    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, xΔt½, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml) 
    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, xΔt½, h, ρ, α, v, c₀, m₀)
    GPUCellListSPH.∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs,  m₀, ρ, c₀, γ, ρ₀) 

    GPUCellListSPH.completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  (0.0, g), gf)

    GPUCellListSPH.update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml) 

end
    #CUDA.registers(@cuda GPUCellListSPH.kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml))


sph_simulation(system, sphkernel, ρ, ρΔt½, v, vΔt½, xΔt½, ∑∂Π∂t, ∑∂ρ∂t, ∑∂v∂t, sumW, sum∇W, ∇Wₙ, Δt, ρ₀, isboundary, ml, h, H⁻¹, m₀, δᵩ, c₀, γ, g, α)

@benchmark  sph_simulation($system, $sphkernel, $ρ, $ρΔt½, $v, $vΔt½, $xΔt½, $∑∂Π∂t, $∑∂ρ∂t, $∑∂v∂t, $sumW, $sum∇W, $∇Wₙ, $Δt, $ρ₀, $isboundary, $ml, $h, $H⁻¹, $m₀, $δᵩ, $c₀, $γ, $g, $α)