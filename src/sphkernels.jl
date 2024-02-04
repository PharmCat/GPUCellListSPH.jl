#=
abstract type  AbstractKernelFunction end

struct CubicSpline <: AbstractKernelFunction end

struct QuarticSpline <: AbstractKernelFunction end

struct QuinticSpline <: AbstractKernelFunction end

struct WendlandC2 <: AbstractKernelFunction end

struct WendlandC4 <: AbstractKernelFunction end

struct WendlandC6 <: AbstractKernelFunction end

function kernelfunction(r, ft::AbstractKernelFunction = WendlandC2())
    kernelfunction(r, ft)
end
function kernelfunction(r, ::CubicSpline)
  max(0, 1 - r)^2 - 4max(0, 1/2 - r)^3
end
function kernelfunction(r, ::QuarticSpline)
    max(0, 1 - r)^4 - 5max(0, 3/5 - r)^4 + 10max(0, 1/5 - r)^4
end
function kernelfunction(r, ::QuinticSpline)
    max(0, 1 - r)^5 - 6max(0, 2/3 - r)^5 + 15max(0, 1/3 - r)^5
end
function kernelfunction(r, ::WendlandC2)
    max(0, 1 - r)^4 * (1 + 4r)
end
function kernelfunction(r, ::WendlandC4)
    max(0, 1 - r)^6 * (1 + 6r + 35/3 * r^2)
    
end
function kernelfunction(r, ::WendlandC6)
    max(0, 1 - r)^8 * (1 + 8r + 25r^2 + 32r^3)
end
=#
