# GPUCellListSPH.jl




```
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

```
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 53 samples with 1 evaluation.
 Range (min … max):  94.191 ms …  96.047 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     94.836 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   94.915 ms ± 362.567 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

            ▁  ▄   ▄  ▁▄█▁ ▁ █     ▄  ▄▁        ▁
  ▆▁▁▆▁▁▁▁▆▁█▆▁█▁▁▆█▆▆████▆█▆█▁▁▆▁▆█▁▁██▁▆▆▁▁▁▁▆█▆▆▁▁▁▁▁▁▁▁▁▁▆ ▁
  94.2 ms         Histogram: frequency by time         95.8 ms <

 Memory estimate: 30.69 KiB, allocs estimate: 519.
```
```