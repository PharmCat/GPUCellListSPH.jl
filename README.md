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
```

## Benchmark

```julia
@benchmark GPUCellListSPH.update!($system)

BenchmarkTools.Trial: 117 samples with 1 evaluation.
 Range (min … max):  42.519 ms …  43.666 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     42.998 ms               ┊ GC (median):    0.00%        
 Time  (mean ± σ):   42.987 ms ± 230.596 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

             ▃     ▃▁  ▁ ▁   ▁ █ ▃▁▃  ▁
  ▇▄▁▁▄▁▆▄▄▁▆█▄▁▆▇▇██▄▇█▆█▇▆▁█▇█▇███▇▆█▄▇▁▆▁▆▁▄▁▆▁▆▆▁▇▁▁▄▁▄▁▄▄ ▄
  42.5 ms         Histogram: frequency by time         43.5 ms <

 Memory estimate: 40.72 KiB, allocs estimate: 722.
```

```julia
@benchmark GPUCellListSPH.partialupdate!($system)

BenchmarkTools.Trial: 118 samples with 1 evaluation.
 Range (min … max):  42.290 ms …  43.137 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     42.673 ms               ┊ GC (median):    0.00%        
 Time  (mean ± σ):   42.672 ms ± 167.919 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                     ▁  ▃  ▄ ▁▆ ▁▁█  ▆▁     ▃
  ▆▆▁▁▁▄▆▁▁▆▇▁▆▁▇▁▆▇▇█▆▄█▁▇█▆██▇███▆▆██▁▆▄▆▄█▆▁▁▄▄▄▇▆▆▁▁▁▁▁▁▁▄ ▄
  42.3 ms         Histogram: frequency by time         43.1 ms <

 Memory estimate: 30.70 KiB, allocs estimate: 509.
```

## Acknowledgment

 * [AhmedSalih3d](https://github.com/AhmedSalih3d)