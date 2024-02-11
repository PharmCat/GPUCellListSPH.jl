using BenchmarkTools, GPUCellListSPH, CUDA, StaticArrays


cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 2)))

cpupoints = map(x->SVector(tuple(x...)), eachrow(rand(Float64, 200000, 2)))

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
    ml  = cu(append!(ones(Float64, size(DF_FLUID, 1)), zeros(Float64, size(DF_BOUND, 1))))

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

    buf     = CUDA.zeros(Float64, N)
    etime = 0.0
#== ==#
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

mx = zeros(Float64, length(x), 2)
for (i, r) in enumerate(eachrow(mx))
    r .= x[i]
end

using ProgressMeter


prog =  Progress(15.75)
val = 0.0
    for i = 1:150
        val += 0.1
        update!(prog, val)
        sleep(0.01)
    end


prog = ProgressThresh(1e-5; desc="Minimizing:")
for val in exp10.(range(2, stop=-6, length=20))
    update!(prog, val)
    sleep(0.1)
end

prog = ProgressUnknown(desc="Burning the midnight oil:", spinner=true)
while true
    next!(prog, spinner="🌑🌒🌓🌔🌕🌖🌗🌘")
    rand(1:10^8) == 0xB00 && break
end
finish!(prog)





minimum(sphprob.∑W)
minimum(x->x[1], sphprob.∑∇W)
minimum(x->x[1], sphprob.∇Wₙ)

minimum(sphprob.∑∂ρ∂t)
minimum(sphprob.∑∂Π∂t)
minimum(sphprob.∑∂v∂t)


minimum(sphprob.ρ)
minimum(sphprob.ρΔt½)


minimum(x->x[1], sphprob.v)
minimum(x->x[1], sphprob.vΔt½)


minimum(x->x[2], sphprob.system.points)
maximum(x->x[2], sphprob.system.points)
minimum(x->x[2], sphprob.xΔt½)


findall(isnan, sphprob.ρ)

sphprob.v[35]
sphprob.system.points[35]
sphprob.∑W[35]
sphprob.∑∇W[35]

p = neighborlist(prob.system)

GPUCellListSPH.∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇Wₙ, p, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml)

prob.∑∂ρ∂t[4197]

prob.∇Wₙ[4197]

prob.xΔt½[4197]

prob.ρ[4197]

prob.v[4197]

kernel_∂ρ∂tDDT!(prob.∑∂ρ∂t,  prob.∇Wₙ, p, prob.xΔt½, prob.h, prob.m₀, prob.δᵩ, prob.c₀, prob.γ, prob.g, prob.ρ₀, prob.ρ, prob.v, prob.ml) 

function kernel_∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter) 

    for index = 1:length(pairs)
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

            #  Δx ⋅ Δx 
            r²    = Δx[1]^2 + Δx[2]^2 
            #=
            z  = Δx[2]
            Cb = (c₀ * c₀ * ρ₀) * γ⁻¹
            Pᴴ =  ρ₀ * g * z
            ρᴴ =  ρ₀ * (((Pᴴ + 1)/Cb)^γ⁻¹ - 1)
            ψ  = 2 * (ρᵢ - ρⱼ) * Δx / r²
            =#
            
            dot3  = -(Δx[1] * ∇Wᵢ[1] + Δx[2] * ∇Wᵢ[2]) #  - Δx ⋅ ∇Wᵢ 
          
            if 1 + DDTgz * Δx[2] < 0 error("!!! $index  $pᵢ   $pⱼ  $Δx $Δv  $∇Wᵢ $r² $dot3" ) end
            drhopvp = ρ₀ * (1 + DDTgz * Δx[2])^γ⁻¹ - ρ₀ ## << CHECK
            
            
            
            visc_densi = DDTkh * c₀ * (ρⱼ - ρᵢ - drhopvp) / (r² + η²)
            delta_i    = visc_densi * dot3 * m₀ / ρⱼ

            drhopvn = ρ₀ * (1 - DDTgz * Δx[2])^γ⁻¹ - ρ₀
            visc_densi = DDTkh * c₀ * (ρᵢ - ρⱼ - drhopvn) / (r² + η²)
            delta_j    = visc_densi * dot3 * m₀ / ρᵢ

            m₀dot     = m₀ * (Δv[1] * ∇Wᵢ[1] + Δv[2] * ∇Wᵢ[2])  #  Δv ⋅ ∇Wᵢ

            ∑∂ρ∂t = (m₀dot + delta_i * MotionLimiter[pᵢ])
            if isnan(∑∂ρ∂t) error("!!! $index  $pᵢ   $pⱼ  $Δx $Δv  $∇Wᵢ $r² $dot3 $drhopvp $visc_densi $delta_i" ) end
            #CUDA.@atomic ∑∂ρ∂t[pⱼ] += (m₀dot + delta_j * MotionLimiter[pⱼ])
            
        end
    end
    return nothing
end