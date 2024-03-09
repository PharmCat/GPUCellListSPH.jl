using BenchmarkTools, GPUCellListSPH, CUDA, StaticArrays


points = map(x->tuple(x...), eachrow(rand(Float64, 200000, 2)))

#cpupoints = map(x->SVector(tuple(x...)), eachrow(rand(Float64, 200000, 2)))
cellsize = (0.04, 0.04, 0.04)
dist = 0.04

dx  = 0.02
h   = 1.2 * sqrt(2) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = H
cellsize = (dist, dist, dist)

points = cpupoints 
if length(points) < 3 error("wrong dimention") end

N = length(first(points))                                          # Number of points 
pcell = CUDA.fill((Int32(0), Int32(0), Int32(0)), N)                  # list of cellst for each particle


MIN    = minimum.(points)                                    # minimal value 
MIN    = @. MIN - abs((MIN + sqrt(eps())) * sqrt(eps()))     # minimal value  (a lillte bit less for better cell fitting)
MAX    = maximum.(points)                                    # maximum                           
range  = MAX .- MIN                                          # range

CELL   = @. ceil(Int, range/cellsize)                        # number of cells 

cellpnum     = CUDA.zeros(Int32, CELL...)                    # array for number of particles in each cell 
cnt          = CUDA.zeros(Int32, 1)                          # temp array for particles counter (need to count place for each pair in pair list)
gpupoints       = CuArray{eltype(first(points))}.(points)                                   # array with particles / points

GPUCellListSPH.cellmap!(pcell, cellpnum, gpupoints,  cellsize, MIN)                 # modify pcell, cellpnum < pcell - map each point to cell, cellpnum - number of particle in each cell

maxpoint = Int(ceil(maximum(cellpnum)*1.05 + 1))                                # mppcell - maximum particle in cell for cell list (with reserve ~ 5%)
mppcell = maxpoint 
    
    
celllist     = CUDA.zeros(Int32, mppcell, CELL...)  

fill!(cellpnum, Int32(0))                                                          # set cell counter to zero 
GPUCellListSPH.fillcells_naive!(celllist, cellpnum,  pcell)  

maxneigh = maximum(cellpnum)*9


mpairs = GPUCellListSPH.мaxpairs(cellpnum)                                                 # mpairs - maximum pairs in pair list (all combination inside cell and neighboring cells (4))

    
    pairs    = CUDA.fill((zero(Int32), zero(Int32)), mpairs)   
    cnt          = CUDA.zeros(Int32, 1)                     # pair list
    GPUCellListSPH.neib_search!(pairs, cnt, cellpnum, gpupoints, celllist, dist)  











using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "../test/input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "../test/input/BoundaryPoints_Dp0.02.csv")
DF_POINTS = append!(CSV.File(fluid_csv) |> DataFrame, CSV.File(boundary_csv) |> DataFrame)
cpupoints = tuple(eachcol(DF_POINTS[!, ["Points:0", "Points:2"]])...)
#cpupoints = tuple(eachcol(Float32.(DF_POINTS[!, ["Points:0", "Points:2"]]))...)
dx  = 0.02
h   = 1.2 * sqrt(2) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = 1.1H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx
α   = 0.01
g   = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (dist, dist)
sphkernel    = WendlandC2(Float64, 2)
system  = GPUCellList(cpupoints, cellsize, dist)
N       = system.n
ρ       = CUDA.zeros(Float64, N)
copyto!(ρ, DF_POINTS.Rhop)
ptype   = CUDA.zeros(Int32, N)
copyto!(ptype, DF_POINTS.ptype)
v       = CUDA.fill((0.0, 0.0), system.n)
sphprob =  SPHProblem(system, dx, h, H, sphkernel, ρ, v, ptype, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)


    GPUCellListSPH.sph∑∇W!(sphprob.∑∇W, sphprob.∇W, sphprob.system.pairs,  sphprob.system.points, sphprob.sphkernel, sphprob.H⁻¹)

    # kernels for each pair
    GPUCellListSPH.sphW!(sphprob.W, sphprob.system.pairs, sphprob.system.points, sphprob.H⁻¹, sphprob.sphkernel)
    # kernels gradientfor each pair
    GPUCellListSPH.sph∇W!(sphprob.∇W,sphprob.system.pairs, sphprob.system.points, sphprob.H⁻¹, sphprob.sphkernel)
    # density derivative with density diffusion
    GPUCellListSPH.∂ρ∂tDDT!(sphprob.∑∂ρ∂t,view(sphprob.system.pairs, 1:sphprob.system.pairsn), sphprob.∇W, sphprob.ρ, sphprob.v, sphprob.system.points, sphprob.h, sphprob.m₀, sphprob.ρ₀, sphprob.c₀, sphprob.γ, sphprob.g, sphprob.δᵩ, sphprob.ptype; minthreads = 256) 
    #  pressure
    GPUCellListSPH.pressure!(sphprob.P, sphprob.ρ, sphprob.c₀, sphprob.γ, sphprob.ρ₀, sphprob.ptype) 
    # momentum equation 
    ∂v∂t!(sphprob.∑∂v∂t,  sphprob.∇W, sphprob.P, view(sphprob.system.pairs, 1:sphprob.system.pairsn),  sphprob.m₀, sphprob.ρ, sphprob.ptype) 
    # add artificial viscosity
    ∂v∂t_av!(sphprob.∑∂v∂t, sphprob.∇W, view(sphprob.system.pairs, 1:sphprob.system.pairsn),  sphprob.system.points, sphprob.h, sphprob.ρ, sphprob.α, sphprob.v, sphprob.c₀, sphprob.m₀, sphprob.ptype)
    # laminar shear stresse
    if sphprob.𝜈 > 0
        ∂v∂t_visc!(sphprob.∑∂v∂t, sphprob.∇W, sphprob.v, sphprob.ρ, sphprob.system.points, view(sphprob.system.pairs, 1:sphprob.system.pairsn), sphprob.h, sphprob.m₀, sphprob.𝜈, sphprob.ptype)
    end
    # add gravity 
    ∂v∂t_addgrav!(sphprob.∑∂v∂t, GPUCellListSPH.gravvec(sphprob.g, sphprob.dim)) 
    #  Boundary forces
    fbmolforce!(sphprob.∑∂v∂t, view(sphprob.system.pairs, 1:sphprob.system.pairsn), sphprob.system.points, 0.4, 2 * sphprob.dx, sphprob.ptype)


    # following steps (update_ρ!, update_vp∂v∂tΔt!, update_xpvΔt!) can be done in one kernel 
        # calc ρ at Δt½
        GPUCellListSPH.update_ρp∂ρ∂tΔt!(sphprob.ρΔt½, sphprob.∑∂ρ∂t, sphprob.Δt * 0.5, sphprob.ρ₀, sphprob.ptype)
        # calc v at Δt½
        GPUCellListSPH.update_vp∂v∂tΔt!(sphprob.vΔt½, sphprob.∑∂v∂t, sphprob.Δt * 0.5, sphprob.ptype) 
        # calc x at Δt½
        GPUCellListSPH.update_xpvΔt!(sphprob.xΔt½, sphprob.vΔt½, sphprob.Δt * 0.5)

        fill!(sphprob.∑∂v∂t[1], zero(T))
        fill!(sphprob.∑∂v∂t[2], zero(T))
        fill!(sphprob.∑∂v∂t[3], zero(T))

        GPUCellListSPH.Δt_stepping(sphprob.buf, sphprob.∑∂v∂t, sphprob.v, sphprob.system.points, sphprob.c₀, sphprob.h, sphprob.CFL, (0,1))


    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t, ∇W, system.pairs, system.points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    
    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇W, system.pairs, system.points, h, ρ, α, v, c₀, m₀) 


    GPUCellListSPH.∂ρ∂tDDT!(sphprob.∑∂ρ∂t,view(sphprob.system.pairs, 1:sphprob.system.pairsn), sphprob.∇W, sphprob.ρ, sphprob.v, sphprob.system.points, sphprob.h, sphprob.m₀, sphprob.ρ₀, sphprob.c₀, sphprob.γ, sphprob.g, sphprob.δᵩ, sphprob.ptype; minthreads = 256)  

    @benchmark  GPUCellListSPH.∂ρ∂tDDT!($sphprob.∑∂ρ∂t,$view(sphprob.system.pairs, 1:sphprob.system.pairsn), $sphprob.∇W, $sphprob.ρ, $sphprob.v, $sphprob.system.points, $sphprob.h, $sphprob.m₀, $sphprob.ρ₀, $sphprob.c₀, $sphprob.γ, $sphprob.g, $sphprob.δᵩ, $sphprob.ptype; minthreads = 256) 
    # 256 - 136.300 / 148.200 / 171.975 μs ±  45.326 μs

    @benchmark GPUCellListSPH.∂Π∂t!($∑∂Π∂t, $∇W, $system.pairs, $system.points, $h, $ρ, $α, $v, $c₀, $m₀; minthreads = 1024) 

    @benchmark GPUCellListSPH.∂ρ∂tDDT_2!($∑∂ρ∂t, $system2.nlist, $system2.cnt, $system2.points, $sphkernel, $h, $H⁻¹, $m₀, $δᵩ, $c₀, $γ, $g, $ρ₀, $ρ, $v, $isboundary) 

#== ==#

prob= sphprob

ρ = copy(prob.ρ)
GPUCellListSPH.cspmcorr!(prob.buf2, prob.W, ρ , prob.m₀, view(prob.system.pairs, 1:prob.system.pairsn), prob.ptype)


#=
function ∂ρ∂tDDT3!(∑∂ρ∂t, buff, ∇W, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024)  where T
    η²    = (0.1*h)*(0.1*h)
    γ⁻¹   = 1/γ
    DDTkh = 2 * h * δᵩ * c₀
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT3!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(∇W)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT3!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∇W)
            Δv    = buff[2][index]
            ∑∂ρ∂t[1][index] = m₀ * (Δv[1] * ∇W[index][1] + Δv[2] * ∇W[index][2])  #  Δv ⋅ ∇Wᵢⱼ
    end
    return nothing
end

function ∂ρ∂tDDT4!(∑∂ρ∂t, buff, ∇W, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024)  where T
    η²    = (0.1*h)*(0.1*h)
    γ⁻¹   = 1/γ
    DDTkh = 2 * h * δᵩ * c₀
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT4!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(∇W)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT4!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if index <= length(∇W)
            Δx    = buff[1][index]
            r²    = Δx[1]^2 + Δx[2]^2 
            ρᵢ    = buff[3][index]
            ρⱼ    = buff[4][index]
            dot3  = -(Δx[1] * ∇W[index][1] + Δx[2] * ∇W[index][2]) #  - Δx ⋅ ∇Wᵢⱼ
            #if ptype[pᵢ] >= 1
                drhopvp = ρ₀ * powfancy7th(1 + DDTgz * Δx[2], γ⁻¹, γ) - ρ₀ 
                visc_densi = DDTkh  * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
            #end
            ∑∂ρ∂t[2][index] = visc_densi * dot3 * m₀ / ρⱼ
    end
    return nothing
end
function ∂ρ∂tDDT5!(∑∂ρ∂t, buff, ∇W, h, m₀, ρ₀, c₀, γ, g, δᵩ, ptype; minthreads::Int = 1024)  where T
    η²    = (0.1*h)*(0.1*h)
    γ⁻¹   = 1/γ
    DDTkh = 2 * h * δᵩ * c₀
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT5!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(∇W)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT5!(∑∂ρ∂t, buff, ∇W,  η², m₀, ρ₀, γ, γ⁻¹, DDTkh, DDTgz, ptype) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(∇W)
            Δx    = buff[1][index]
            r²    = Δx[1]^2 + Δx[2]^2 
            ρᵢ    = buff[3][index]
            ρⱼ    = buff[4][index]
            dot3  = -(Δx[1] * ∇W[index][1] + Δx[2] * ∇W[index][2]) #  - Δx ⋅ ∇Wᵢⱼ
            #if ptype[pⱼ] >= 1
                drhopvn = ρ₀ * powfancy7th(1 - DDTgz * Δx[2], γ⁻¹, γ) - ρ₀
                visc_densi = DDTkh  * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            #end
            ∑∂ρ∂t[3][index] = visc_densi * dot3 * m₀ / ρᵢ
    end
    return nothing
end


function collect_∂ρ∂tDDT2!(∑∂ρ∂t::CuArray{T}, buff, pairs, ptype; minthreads::Int = 1024)  where T
    fill!(∑∂ρ∂t, zero(T))
    gpukernel = @cuda launch=false kernel_collect_∂ρ∂tDDT2!(∑∂ρ∂t, buff, pairs, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, buff, pairs, ptype; threads = Tx, blocks = Bx)
end
function kernel_collect_∂ρ∂tDDT2!(∑∂ρ∂t, buff, pairs, ptype) 
    tindex = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    index      = tindex
    while index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if ptype[pᵢ] > 0 && ptype[pⱼ] > 0
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += buff[1][index] + buff[2][index]
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += buff[1][index] + buff[3][index]
        end
        index += stride
    end
    return nothing
end
=#



function pairs_calk_test2!(buff) 
    gpukernel = @cuda launch=false kernel_pairs_calk_test2!(buff) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buff)
    maxThreads = config.threads
    Tx  = min( maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk_test2!(buff) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buff)
        pair = buff[index]
        a = pair.ρᵢ + pair.ρⱼ + pair.Δx[1] + pair.Δv[1]
    end
    return nothing
end
function pairs_calk_test1!(buff) 
    gpukernel = @cuda launch=false kernel_pairs_calk_test1!(buff) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buff)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk_test1!(buff) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buff)
        Δx = buff[1][index] 
        Δv = buff[2][index] 
        ρᵢ = buff[3][index] 
        ρⱼ = buff[4][index] 
        a = ρᵢ + ρⱼ + Δx[1] + Δv[2]
    end
    return nothing
end


@benchmark pairs_calk_test1!($buff) 

@benchmark pairs_calk_test2!($buffpp) 






















function sph_simulation(system, sphkernel, ρ, ρΔt½, v, vΔt½, xΔt½, ∑∂Π∂t, ∑∂ρ∂t, ∑∂v∂t, sumW, sum∇W, ∇Wₙ, Δt, ρ₀, isboundary, ml, h, H⁻¹, m₀, δᵩ, c₀, γ, g, α; simn = 1)

    for iter = 1:simn
    GPUCellListSPH.update!(system)
    x     = system.points
    pairs = system.pairs

    fill!(sumW, zero(Float64))
    fill!(∑∂ρ∂t, zero(Float64))
    fill!(∑∂Π∂t, zero(Float64))
    fill!(∑∂v∂t, zero(Float64))

    if length(∇Wₙ) != length(pairs)
        CUDA.unsafe_free!(∇Wₙ)
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

    GPUCellListSPH.∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, xΔt½, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρΔt½, vΔt½, ml) 
    GPUCellListSPH.∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, xΔt½, h, ρΔt½, α, vΔt½, c₀, m₀)
    GPUCellListSPH.∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs,  m₀, ρΔt½, c₀, γ, ρ₀) 

    GPUCellListSPH.completed_∂v∂t!(∑∂v∂t, ∑∂Π∂t,  (0.0, g), gf)

    GPUCellListSPH.update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml)
    
    etime += Δt
    Δt = GPUCellListSPH.Δt_stepping(buf, ∑∂v∂t, v, x, c₀, h, CFL)
    end

    #GPUCellListSPH.create_vtp_file(joinpath(path, "./input/OUTPUT.vtk"), x, ρ, ∑∂v∂t, v, etime)
end
    #CUDA.registers(@cuda GPUCellListSPH.kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, cellcounter, pairs, gpupoints, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ml))


sph_simulation(system, sphkernel, ρ, ρΔt½, v, vΔt½, xΔt½, ∑∂Π∂t, ∑∂ρ∂t, ∑∂v∂t, sumW, sum∇W, ∇Wₙ, Δt, ρ₀, isboundary, ml, h, H⁻¹, m₀, δᵩ, c₀, γ, g, α)

@benchmark  sph_simulation($system, $sphkernel, $ρ, $ρΔt½, $v, $vΔt½, $xΔt½, $∑∂Π∂t, $∑∂ρ∂t, $∑∂v∂t, $sumW, $sum∇W, $∇Wₙ, $Δt, $ρ₀, $isboundary, $ml, $h, $H⁻¹, $m₀, $δᵩ, $c₀, $γ, $g, $α; simn = 100)




using GPUCellListSPH
using CSV, DataFrames, CUDA, BenchmarkTools
using SPHKernels, WriteVTK
path         = dirname(@__FILE__)
fluid_csv    = joinpath(path, "./input/FluidPoints_Dp0.02.csv")
boundary_csv = joinpath(path, "./input/BoundaryPoints_Dp0.02.csv")

cpupoints, DF_FLUID, DF_BOUND    = GPUCellListSPH.loadparticles(fluid_csv, boundary_csv)

dx  = 0.02
h   = 1.2 * sqrt(2) * dx
H   = 2h
h⁻¹ = 1/h
H⁻¹ = 1/H
dist = H
ρ₀  = 1000.0
m₀  = ρ₀ * dx * dx #mᵢ  = mⱼ = m₀
α   = 0.01
g   = 9.81
c₀  = sqrt(g * 2) * 20
γ   = 7
Δt  = dt  = 1e-5
δᵩ  = 0.1
CFL = 0.2
cellsize = (H, H)
sphkernel    = WendlandC2(Float64, 2)

system  = GPUCellListSPH.GPUCellList(cpupoints, cellsize, H)

ρ           = cu(Array([DF_FLUID.Rhop;DF_BOUND.Rhop]))
ml          = cu(append!(ones(Float64, size(DF_FLUID, 1)), zeros(Float64, size(DF_BOUND, 1))))
isboundary  = .!Bool.(ml)
gf          = cu([-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))])
v           = CUDA.fill((0.0, 0.0), length(cpupoints))


sphprob =  SPHProblem(system, h, H, sphkernel, ρ, v, ml, gf, isboundary, ρ₀, m₀, Δt, α, g, c₀, γ, δᵩ, CFL)

stepsolve!(sphprob, 1)


get_points(sphprob)

get_velocity(sphprob)

get_density(sphprob)

get_acceleration(sphprob)


@benchmark stepsolve!($sphprob, 1)

#=
BenchmarkTools.Trial: 946 samples with 1 evaluation.
 Range (min … max):  4.714 ms … 42.996 ms  ┊ GC (min … max): 0.00% … 54.74%
 Time  (median):     5.193 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.284 ms ±  1.250 ms  ┊ GC (mean ± σ):  0.47% ±  1.78%

               ▁▃▄▄█▅▆▄▅▂▃▃▁▁
  ▂▁▁▂▂▂▃▄▄▄▄▇▇███████████████▇▅▅▅▄▄▄▄▃▄▄▃▃▄▄▄▃▃▃▂▃▃▃▂▂▂▃▃▃▂ ▄
  4.71 ms        Histogram: frequency by time        6.04 ms <

 Memory estimate: 100.20 KiB, allocs estimate: 1938.
=#

#=
findfirst(x-> (x[1] == list[1][1] &&  x[2] == list[1][2]) || (x[2] == list[1][1] &&  x[1] == list[1][2]), Array(pairs))


function Δt_test(α, points, v, c₀, h, CFL)
    eta2  = (0.01)h * (0.01)h
    visc  = maximum(@. abs(h * dot(v, points) / (dot(points, points) + eta2)))
    println("visc ", visc)
    dt1   = minimum(@. sqrt(h / norm(α)))
    println("dt1 ", dt1)
    dt2   = h / (c₀ + visc)
    println("dt2 ",dt2)
    dt    = CFL * min(dt1, dt2)

    return dt
end

Δt_test(acceleration, points, velocity, c₀, h, CFL)

function kernel_Δt_stepping!(buf, v, points, h, η²) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        vp = v[index]
        pp = points[index]
        buf[index] = abs(h * (vp[1] * pp[1] + vp[2] * pp[2]) / (pp[1]^2 + pp[2]^2 + η²))
    end
    return nothing
end
function kernel_Δt_stepping_norm!(buf, a) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buf)
        buf[index] =  sqrt(a[index, 1]^2 + a[index, 2]^2) 
    end
    return nothing
end
"""    
    Δt_stepping(buf, a, v, points, c₀, h, CFL) 

"""
function Δt_stepping_test(buf, a, v, points, c₀, h, CFL) 
    η²  = (0.01)h * (0.01)h

    gpukernel = @cuda launch=false kernel_Δt_stepping_norm!(buf, a) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buf)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(buf, a; threads = Tx, blocks = Bx)

    dt1 = sqrt(h / maximum(buf))
    println("dt1 ", dt1)

    gpukernel = @cuda launch=false kernel_Δt_stepping!(buf, v, points, h, η²) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(buf)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx = cld(Nx, Tx)
    CUDA.@sync gpukernel(buf, v, points, h, η²; threads = Tx, blocks = Bx)
    
    visc  = maximum(buf)

    println("visc ", visc)

    dt2   = h / (c₀ + visc)
    println("dt2 ",dt2)
    dt    = CFL * min(dt1, dt2)

    return dt
end

Δt_stepping_test(buf, ∑∂v∂t, v, x, c₀, h, CFL) 
=#
#=

function ∂Πᵢⱼ∂t(list, points, h, ρ, α, v, c₀, m₀, WgL)
    N    = length(points)

    η²    = (0.1 * h) * (0.1 * h)

    iter = [1]
    L  = list[1]
        i = L[1]; j = L[2];
        
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        xᵢⱼ   = points[i] - points[j]
        ρᵢⱼ   = (ρᵢ + ρⱼ) * 0.5

        cond      = dot(vᵢⱼ, xᵢⱼ)

        cond_bool = cond < 0

        μᵢⱼ = h * cond / (dot(xᵢⱼ, xᵢⱼ) + η²)
        Πᵢⱼ = cond_bool * (-α * c₀ * μᵢⱼ) / ρᵢⱼ
        
        Πᵢⱼm₀WgLi = Πᵢⱼ * m₀ * WgL[iter]
        
        viscIi   = -Πᵢⱼm₀WgLi
        viscIj   =  Πᵢⱼm₀WgLi

    return viscIi, viscIj
end

=#




#=
function ∂ρᵢ∂tDDTtest(list, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, WgL, MotionLimiter)

    η²   = (0.1*h)*(0.1*h)

    iter = 1
    L    = list[1]
        i = L[1]; j = L[2];

        xᵢⱼ   = points[i] - points[j]
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]

        Cb    = (c₀^2 * ρ₀) / γ

        r²    = dot(xᵢⱼ, xᵢⱼ)
        println(xᵢⱼ)
        println(r²)

        DDTgz = ρ₀ * g / Cb
        DDTkh = 2 * h * δᵩ

        dot3  = -dot(xᵢⱼ, ∇ᵢWᵢⱼ)

        println(dot3)

        # Do note that in a lot of papers they write "ij"
        # BUT it should be ji for the direction to match (in dot3)
        # the density direction
        # For particle i
        drz   = xᵢⱼ[2]             # 
        rh    = 1 + DDTgz * drz
        
        drhop = ρ₀* ^(rh, 1 / γ) - ρ₀   # drhop = ρ₀* (rh^invγ  - 1)
        println(drhop)
        visc_densi = DDTkh * c₀ *(ρⱼ - ρᵢ - drhop) / (r² + η²)
        println(visc_densi)
        delta_i = visc_densi * dot3 * m₀ / ρⱼ
        println(delta_i)

        # For particle j
        drz   = -xᵢⱼ[2]
        rh    = 1 + DDTgz * drz
        drhop = ρ₀* ^(rh, 1/γ) - ρ₀
        visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhop) / (r² + η²)
        
        delta_j = visc_densi * dot3 * m₀ / ρᵢ

        m₀dot     = m₀ * dot(vᵢⱼ, ∇ᵢWᵢⱼ) 

        dρdtIi = m₀dot + delta_i * MotionLimiter[i]
        dρdtIj = m₀dot + delta_j * MotionLimiter[j]

        dρdtLi = m₀dot + delta_i * MotionLimiter[i]

    return dρdtIi, dρdtIj, dρdtLi
end

∂ρᵢ∂tDDTtest(list, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, density, velocity, WgL, MotionLimiter)


∇Wₙ[18631]
WgL[1]

function kernel_∂ρ∂tDDT_test!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 
    index = 18631
    if index <= length(pairs)
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]; d = pair[3]
        if !isnan(d)

            γ⁻¹  = 1/γ
            η²   = (0.1*h)*(0.1*h)
            Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
            DDTgz = ρ₀ * g / Cb
            DDTkh = 2 * h * δᵩ
    
            #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
            =#

            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇Wᵢ   = ∇Wₙ[index]

            #r²    = (xᵢ[1]-xⱼ[1])^2 + (xᵢ[2]-xⱼ[2])^2
            r² = d^2  #  xᵢ⋅ xⱼ = d^2
            println(r²)
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            
            dot3  = -(Δx[1] * ∇Wᵢ[1] + Δx[2] * ∇Wᵢ[2]) #  - Δx ⋅ ∇Wᵢ 
            println(dot3)
            
            drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀
            
            visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
            
            delta_i    = visc_densi * dot3 * m₀ / ρⱼ

            drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
            println(drhopvn)

            visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            println("DDTkh =" , DDTkh, "; visc_densi =", visc_densi)
            delta_j    = visc_densi * dot3 * m₀ / ρᵢ
            println(delta_i)
            m₀dot     = m₀ * (Δv[1] * ∇Wᵢ[1] + Δv[2] * ∇Wᵢ[2])  #  Δv ⋅ ∇Wᵢ

            ∑∂ρ∂ti = (m₀dot + delta_i * MotionLimiter[pᵢ])
            ∑∂ρ∂tj = (m₀dot + delta_j * MotionLimiter[pⱼ])
            
        end
    end
    ∑∂ρ∂ti, ∑∂ρ∂tj
end

kernel_∂ρ∂tDDT_test!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 
=#


#=
        using CUDA

        function test!(x) 
            gpukernel = @cuda launch=false kernel_test!(x) 
            config = launch_configuration(gpukernel.fun)
            Nx = length(x)
            maxThreads = config.threads
            #maxBlocks  = config.blocks
            Tx  = min(maxThreads, Nx)
            #Bx  = min(maxBlocks, cld(Nx, Tx))
            Bx  = cld(Nx, Tx)
            CUDA.@sync gpukernel(x; threads = Tx, blocks = Bx)
        end
        function kernel_test!(x) 
            index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
            stride = gridDim().x * blockDim().x
            i = index
            n = 1
            while  i <= length(x)
                x[i] = n
                i += stride
                n += 1
            end
            return nothing
        end
        x = CUDA.zeros(1000000000)
        test!(x) 
        sum(x)


        function test2!(x) 
            gpukernel = @cuda launch=false kernel_test2!(x) 
            config = launch_configuration(gpukernel.fun)
            Nx = length(x)
            maxThreads = config.threads
            maxThreads = 3
            Tx  = min(maxThreads, Nx)
            CUDA.@sync gpukernel(x; threads = Tx, blocks = 1)
        end
        function kernel_test2!(x) 
            index  = threadIdx().x
            stride = blockDim().x
            i = index
            while i <= length(x)
                @cuprintln "i = $i, index = $index, threadIdx: $(threadIdx().x), blockIdx $(blockIdx().x), blockDim $(blockDim().x)" 
                i += stride
            end
            return nothing
        end
        test2!(CUDA.zeros(10)) 

        @benchmark minmax(34, 23)


using CUDA

        function test_2d!(cnt, mat)
            gpukernel = @cuda launch=false kernel_test_2d!(cnt, mat, 6)
            config = launch_configuration(gpukernel.fun)
            maxThreads = config.threads
            Nx = length(mat)
            Tx  = min(maxThreads, Nx) 
            Bx  = 1
            cs = fld(attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) - 8,  Tx * sizeof(Tuple{Int32, Int32}))
            CUDA.@sync gpukernel(cnt, mat, cs; threads = Tx, blocks = Bx, shmem= Tx * cs * sizeof(Tuple{Int32, Int32}))
        end
        function kernel_test_2d!(cnt, mat, cs)
            index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
            stride = gridDim().x * blockDim().x
            scnt   = CuStaticSharedArray(Int32, 1)
     
            if threadIdx().x == 1
                scnt[1] = cnt[1] 
            end
            sync_threads()
            Nx, Ny = size(mat)
            while index <= length(mat)
                y = cld(index, Nx)
                x = index - Nx * (y - 1)
                #@cuprintln "index $index ind $x, $y"

                n = CUDA.@atomic scnt[1] += 1
                mat[x,y] = n + 1
                index += stride
            end
            sync_threads()
            if threadIdx().x == 1
                cnt[1] = scnt[1] 
                CUDA.@cuprintln "Down $(cnt[1])  block $(blockIdx().x)"
            end
            return nothing
        end
        cnt = cu([0])
        mat = CUDA.zeros(100, 100)
        test_2d!(cnt, mat)



        function pranges_test!(ranges, pairs) 
            gpukernel = @cuda launch=false kernel_pranges_test!(ranges, pairs) 
            config = launch_configuration(gpukernel.fun)
            Nx = length(ranges)
            maxThreads = config.threads
            Tx  = min(maxThreads, Nx)
            CUDA.@sync gpukernel(ranges, pairs; threads = 1, blocks = 1, shmem = Tx * sizeof(Int))
        end
        function kernel_pranges_test!(ranges, pairs, np) 
            index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
            thread = threadIdx().x
            stride = gridDim().x * blockDim().x
            tn     = blockDim().x
            cache  = CuDynamicSharedArray(Int, tn)
            cache[thread] = 1

            si = (thread - 1) * np + 1
            ei = min(length(pairs), thread * np)
            sync_threads()
            if thread != 1
                s = first(pairs[si])
                for i = si+1:length(pairs)
                    if first(pairs[i]) != s
                        si = i
                        break
                    end
                end
            end 
      
            sync_threads()            
            for i = si:ei
        
            end
        end


        @benchmark  pranges_test!($pr, $system.pairs)

=#
#=
function kernel_neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(points)
        # get point cell
        cell   = pcell[index]
        celli  = cell[1] + offset[1]
        cellj  = cell[2] + offset[2]
        if  0 < celli <= size(celllist, 2) && 0 < cellj <= size(celllist, 3)
            clist  = view(celllist, :, celli, cellj)
            celln  = cellpnum[celli, cellj]
            distsq = dist^2
            cnt    = ncnt[index]
            for i = 1:celln
                indexj = clist[i]
                if index != indexj && (points[index][1] - points[indexj][1])^2 + (points[index][2] - points[indexj][2])^2 < distsq
                    cnt += 1
                    if cnt <= size(nlist, 1)
                        nlist[cnt, index] = indexj
                    end
                end
            end
            ncnt[index] = cnt
        end
    end
    return nothing
end
"""
    neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)

"""
function neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)
    gpukernel = @cuda launch=false kernel_neiblist_2d!(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset)
    config = launch_configuration(gpukernel.fun)
    Nx = length(points)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(nlist, ncnt, points,  celllist, cellpnum, pcell, dist, offset; threads = Tx, blocks = Bx)
end
=#
#=
function ∂ρ∂tDDT2!(∑∂ρ∂t,  ∇W, pairs, points, ranges, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    if length(pairs) != length(∇W) error("Length shoul be equal") end

    gpukernel = @cuda launch=false kernel_∂ρ∂tDDT2!(∑∂ρ∂t,  ∇W, pairs, points, ranges, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(ranges)
    maxThreads = config.threads
    Tx  = min(maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(∑∂ρ∂t, ∇W, pairs, points, ranges, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype; threads = Tx, blocks = Bx)
end
function kernel_∂ρ∂tDDT2!(∑∂ρ∂t,  ∇W, pairs, points, ranges, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype) 
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    #s      = index * (stride - 1) + index
    #e      = stride - 1
    # move it outside kernel
    γ⁻¹  = 1/γ
    η²   = (0.1*h)*(0.1*h)
    Cb    = (c₀ * c₀ * ρ₀) * γ⁻¹
    DDTgz = ρ₀ * g / Cb
    DDTkh = 2 * h * δᵩ

    while index <= length(ranges)
        s, e = ranges[index]
        for pind in s:e
        pair  = pairs[index]
        pᵢ    = pair[1]; pⱼ = pair[2]
        if pᵢ > 0 # && !(isboundary[pᵢ] && isboundary[pᵢ]) 
            xᵢ    = points[pᵢ]
            xⱼ    = points[pⱼ]
            Δx    = (xᵢ[1] - xⱼ[1], xᵢ[2] - xⱼ[2])
            r²    = Δx[1]^2 + Δx[2]^2 
            # for timestep Δt½ d != actual range
            # one way - not calculate values out of 2h
            # if r² > (2h)^2 return nothing end
            #=
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ᵸᵀᴴ
            =#
            ρᵢ    = ρ[pᵢ]
            ρⱼ    = ρ[pⱼ]

            Δv    = (v[pᵢ][1] - v[pⱼ][1], v[pᵢ][2] - v[pⱼ][2])

            ∇Wᵢⱼ  = ∇W[index]
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            dot3  = -(Δx[1] * ∇Wᵢⱼ[1] + Δx[2] * ∇Wᵢⱼ[2]) #  - Δx ⋅ ∇Wᵢⱼ

            # as actual range at timestep Δt½  may be greateg  - some problems can be here
            if 1 + DDTgz * Δx[2] < 0 || 1 - DDTgz * Δx[2] < 0 return nothing end
            
            m₀dot     = m₀ * (Δv[1] * ∇Wᵢⱼ[1] + Δv[2] * ∇Wᵢⱼ[2])  #  Δv ⋅ ∇Wᵢⱼ
            ∑∂ρ∂ti = ∑∂ρ∂tj = m₀dot

            if ptype[pᵢ] >= 1
                drhopvp = ρ₀ * powfancy7th(1 + DDTgz * Δx[2], γ⁻¹, γ) - ρ₀ ## << CHECK
                visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
                delta_i    = visc_densi * dot3 * m₀ / ρⱼ
                ∑∂ρ∂ti    += delta_i 
            end
            CUDA.@atomic ∑∂ρ∂t[pᵢ] += ∑∂ρ∂ti 

            if ptype[pⱼ] >= 1
                drhopvn = ρ₀ * powfancy7th(1 - DDTgz * Δx[2], γ⁻¹, γ) - ρ₀
                visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
                delta_j    = visc_densi * dot3 * m₀ / ρᵢ
                ∑∂ρ∂tj    += delta_j 
            end
            CUDA.@atomic ∑∂ρ∂t[pⱼ] += ∑∂ρ∂tj
            
            #=
            if isnan(delta_j) || isnan(m₀dot)  || isnan(ρᵢ) || isnan(ρⱼ) 
                @cuprintln "kernel_DDT 1 isnan dx1 = $(Δx[1]) , dx2 = $(Δx[2]) rhoi = $ρᵢ , dot3 = $dot3 , visc_densi = $visc_densi drhopvn = $drhopvn $(∇W[1]) $(Δv[1])"
                error() 
            end
            if isinf(delta_j) || isinf(m₀dot)  || isinf(delta_i) 
                @cuprintln "kernel_DDT 2 inf: dx1 = $(Δx[1]) , dx2 = $(Δx[2]) rhoi = $ρᵢ , rhoj = $ρⱼ , dot3 = $dot3 ,  delta_i = $delta_i , delta_j = $delta_j , drhopvn = $drhopvn , visc_densi = $visc_densi , $(∇W[1]) , $(Δv[1])"
                error() 
            end
            =#
            #mlfac = MotionLimiter[pᵢ] * MotionLimiter[pⱼ]
            #=
            if isnan(∑∂ρ∂tval1) || isnan(∑∂ρ∂tval2) || abs(∑∂ρ∂tval1) >  10000000 || abs(∑∂ρ∂tval2) >  10000000
                @cuprintln "kernel DDT: drhodti = $∑∂ρ∂ti drhodtj = $∑∂ρ∂tj, dx1 = $(Δx[1]), dx2 = $(Δx[2]) rhoi = $ρᵢ, rhoj = $ρⱼ, dot3 = $dot3, visc_densi = $visc_densi, drhopvn = $drhopvn, dw = $(∇W[1]),  dv = $(Δv[1])"
                error() 
            end
            =#
            
        end
        index += stride
        end
    end
    return nothing
end
=#


struct MyStr{T}
    Δxˣ::T
    Δxʸ::T
    Δvˣ::T
    Δvʸ::T
    ρᵢ::T
    ρⱼ::T
end

function pairs_calk!(buff, pairs; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false kernel_pairs_calk!(buff, pairs) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff, pairs; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk!(buff, pairs) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        buff[1][index] = 1.2
        buff[2][index] = 4.5
        buff[3][index] = 0.1
        buff[4][index] = 0.7
        buff[5][index] = 0.7
        buff[6][index] = 0.7
        buff[7][index] = 0.7
        buff[8][index] = 0.7
    end
    return nothing
end
function pairs_calk2!(buff, pairs; minthreads::Int = 1024)  
    gpukernel = @cuda launch=false maxregs=64 kernel_pairs_calk2!(buff, pairs) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff, pairs; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk2!(buff::AbstractArray{MyStr{T}}, pairs)  where T
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(buff)
        buff[index] = MyStr{T}(0.1, 0.2, 0.3, 0.4, 0.9, 1.2)
    end
    return nothing
end

FT = Float32
PN = 10000
buff = (CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN), CUDA.zeros(FT, PN))

prs = CUDA.zeros(PN )
@benchmark pairs_calk!($buff,$prs ; minthreads = 1024)

buff2 = CuArray{MyStr{FT}}(undef, PN)

@benchmark pairs_calk2!($buff2,$pairs ; minthreads = 1024)

zv = @SVector zeros(8)
buff3 = CUDA.fill(zv, PN)
function pairs_calk3!(buff, pairs; minthreads::Int = 1024) 
    gpukernel = @cuda launch=false kernel_pairs_calk3!(buff, pairs) 
    config = launch_configuration(gpukernel.fun)
    Nx = length(pairs)
    maxThreads = config.threads
    Tx  = min(minthreads, maxThreads, Nx)
    Bx  = cld(Nx, Tx)
    CUDA.@sync gpukernel(buff, pairs; threads = Tx, blocks = Bx)
end
function kernel_pairs_calk3!(buff, pairs) 
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if index <= length(pairs)
        buff[index] = SVector((0.1, 0.2, 0.3, 0.4, 0.9, 1.2, 4.5, 7.8))
    end
    return nothing
end
@benchmark pairs_calk3!($buff3,$prs ; minthreads = 1024)