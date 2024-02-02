module GPUCellListSPH

using CUDA

export GPUCellList, update!

include("kernels.jl")
include("structs.jl")

end
