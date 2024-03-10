module GPUCellListSPH

import Base: show

using CUDA, SPHKernels, CSV, DataFrames, WriteVTK, ProgressMeter, StaticArrays, Plots, Roots, LinearAlgebra

import Plots: Animation

export GPUCellList, update!, partialupdate!, neighborlist, neighborlistview

export makedf, writecsv

export SPHProblem, stepsolve!, timesolve!, get_points, get_velocity, get_density, get_pressure, get_acceleration

export sphW!, sph∇W!, sph∑∇W!, sph∑∇W!

export ∂ρ∂tDDT!, pressure!, ∂v∂t!,  ∂v∂t_av!, ∂v∂t_visc!, ∂v∂t_addgrav!, ∂v∂tpF!, dpcreg!, cspmcorr!, xsphcorr!, fbmolforce!

#include("sphkernels.jl")
include("gpukernels_cls.jl")      # GPU functions for 2d case
include("gpukernels_sph.jl")      # GPU functions for 3d case
#include("gpulistkernels.jl")
include("celllist.jl")          # Cell List Neighbors Search for GPU
#include("gpunlist.jl")
include("sphproblem.jl")        # main SPH solver
include("writevtk.jl")          # write VTK files
#include("egpukernels.jl")       # reorganized experimental SPH solver
include("auxillary.jl")         # loadparticles
end
