# GPUCellListSPH.jl




```julia
using CUDA, BenchmarkTools

cpupoints = map(x->tuple(x...), eachrow(rand(Float64, 200000, 3)))


system = GPUCellListSPH.GPUCellList(cpupoints, (0.016, 0.016), 0.016)

system.points # points

system.pairs # pairs for each cell

system.grid # cell grid 

sum(system.cellpnum) # total cell number

maximum(system.cellpnum) # maximum particle in cell

maximum(system.cellcounter) # maximum pairs in cell

count(x-> !isnan(x[3]), system.pairs) == sum(system.cellcounter)

GPUCellListSPH.update!(system) # update the system

count(x-> !isnan(x[3]), system.pairs) == sum(system.cellcounter)

@benchmark GPUCellListSPH.update!($system)

```julia
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 108 samples with 1 evaluation.
 Range (min … max):  46.203 ms …  47.297 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     46.593 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   46.616 ms ± 184.553 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▂  ▂▂ ▆ ▆ █▂▄▄▄▄▂   ▂
  ▆▁▁▁▁▁▁▁█▄▁▄█▄▄██▆███▆███████▄████▄▆▆▁▁▄▆▆▄▁▆▁▁▁▁▁▆▁▄▁▁▁▁▄▁▄ ▄
  46.2 ms         Histogram: frequency by time         47.2 ms <

 Memory estimate: 29.17 KiB, allocs estimate: 485.
```