# GPUCellListSPH.jl

## Description

This is concept-project for particle cell neiborhood list and SPH. Based on [AhmedSalih3d](https://github.com/AhmedSalih3d) repo [SPHExample](https://github.com/AhmedSalih3d/SPHExample).


## Install

```
import Pkg
Pkg.add(url = "https://github.com/PharmCat/GPUCellListSPH.jl.git")
```

## Using 

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
```

## Benchmark

```julia
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 120 samples with 1 evaluation.
 Range (min … max):  40.410 ms …  43.864 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     41.954 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   41.970 ms ± 739.440 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁                ██▃▁ ▃  ▁     █▁▃   ▃  ▃    ▃
  █▄▄▆▁▄▁▄▁▄▁▁▆▄▁▇▆████▄█▄▆█▆▇▄▁▇███▆▇▆█▇▆█▁▁▄▆█▄▄▄▄▁▁▁▁▁▁▁▁▄▄ ▄
  40.4 ms         Histogram: frequency by time         43.8 ms <

 Memory estimate: 33.23 KiB, allocs estimate: 515.
```

```julia
@benchmark GPUCellListSPH.partialupdate!($system)

BenchmarkTools.Trial: 122 samples with 1 evaluation.
 Range (min … max):  40.010 ms …  42.587 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     41.097 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   41.185 ms ± 491.277 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                      ▄▇█▁▁▇▄▅▁▂
  ▃▁▃▃▁▃▁▃▃▁▃▁▆▃▁▁▁▃█▅██████████▃▅▃▃▆▁▃▃▃▆▃▃▁▃▃▃▅▆▅▃▃▃▃▁▁▁▃▁▁▃ ▃
  40 ms           Histogram: frequency by time         42.5 ms <

 Memory estimate: 25.39 KiB, allocs estimate: 400.
```

## Acknowledgment

 * [AhmedSalih3d](https://github.com/AhmedSalih3d)