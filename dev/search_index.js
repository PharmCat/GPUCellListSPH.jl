var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [GPUCellListSPH]","category":"page"},{"location":"api/#GPUCellListSPH.GPUCellList-Union{Tuple{T}, Tuple{AbstractArray{<:Tuple{Vararg{T}}}, Any, T}} where T<:AbstractFloat","page":"API","title":"GPUCellListSPH.GPUCellList","text":"GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)\n\nMake cell list structure.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.SPHProblem","page":"API","title":"GPUCellListSPH.SPHProblem","text":"SPHProblem(system::GPUCellList, h::Float64, H::Float64, sphkernel::AbstractSPHKernel, ρ, v, ptype, ρ₀::Float64, m₀::Float64, Δt::Float64, α::Float64, g::Float64, c₀::Float64, γ, δᵩ::Float64, CFL::Float64; s::Float64 = 0.0)\n\nSPH simulation data structure.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPUCellListSPH.W_2d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.W_2d!","text":"W_2d!(W, pairs, sphkernel, H⁻¹)\n\nCompute kernel values for each particles pair in list. Update W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.cellmap_2d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.cellmap_2d!","text":"cellmap_2d!(pcell, cellpnum, points,  h, offset)\n\nMap each point to cell and count number of points in each cell.\n\nFor each coordinates cell number calculated:\n\ncsᵢ = size(cellpnum, 1) \np₁  =  (x₁ - offset₁) * h₁⁻¹\npᵢ₁ = ceil(min(max(p₁, 1), csᵢ))\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.cellmap_3d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.cellmap_3d!","text":"cellmap_3d!(pcell, cellpnum, points,  h, offset)\n\nMap each point to cell and count number of points in each cell.\n\nFor each coordinates cell number calculated:\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.completed_∂v∂t!-Tuple{Any, Any, Any}","page":"API","title":"GPUCellListSPH.completed_∂v∂t!","text":"completed_∂vᵢ∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac)\n\nAdd gravity and artificial viscosity to the momentum equation.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.completed_∂v∂t_3d!-Tuple{Any, Any, Any}","page":"API","title":"GPUCellListSPH.completed_∂v∂t_3d!","text":"completed_∂vᵢ∂t!(∑∂v∂t, ∑∂Π∂t,  gvec, gfac)\n\nAdd gravity and artificial viscosity to the momentum equation.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.cspmcorr!-NTuple{8, Any}","page":"API","title":"GPUCellListSPH.cspmcorr!","text":"cspmcorr!(∑ρcspm1, ∑ρcspm2, ρ, m₀, pairs, points, sphkernel, H⁻¹)\n\nCorrected Smoothed Particle Method (CSPM) Density Renormalisation.\n\nChen JK, Beraun JE, Carney TC (1999) A corrective smoothed particle method for boundary value problems in heat conduction. Int. J. Num. Meth. Engng. https://doi.org/10.1002/(SICI)1097-0207(19990920)46:2<231::AID-NME672>3.0.CO;2-K\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.dpcreg!-NTuple{13, Any}","page":"API","title":"GPUCellListSPH.dpcreg!","text":"dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim)\n\nDynamic Particle Collision (DPC) correction.\n\ndelta textbfv_i^DPC = sum k_ijfracm_jm_i + m_jv_ij^coll + fracDelta  trho_isum phi_ij frac2V_jV_i + V_jfracp_ij^br_ij^2 + eta^2textbfr_ij\n\n\n\n(v_ij^coll  quad phi_ij) = begincases (fractextbfv_ijcdot textbfr_ijr_ij^2 + eta^2textbfr_ji quad 0)  textbfv_ijcdot textbfr_ij  0  (0 quad 1)   otherwise endcases\n\n\np_ij^b = tildep_ij chi_ij \n\n\n\ntildep_ij = max(min(lambda p_i + p_j lambda p_max) p_min)\n\n\n\nchi_ij  = sqrtfracomega(r_ij l_0)omega(l_02 l_0)\n\n\n\nk_ij =  begincases chi_ij  05 le r_ijl_0  1  1  r_ijl_0  05 endcases\n\n\nMojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.dpcreg_3d!-NTuple{13, Any}","page":"API","title":"GPUCellListSPH.dpcreg_3d!","text":"dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim)\n\nDynamic Particle Collision (DPC) correction.\n\nMojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.fillcells_naive_2d!-Tuple{Any, Any, Any}","page":"API","title":"GPUCellListSPH.fillcells_naive_2d!","text":"fillcells_naive_2d!(celllist, cellpnum, pcell)\n\nFill cell list with cell. Naive approach. No bound check. Values in pcell list shoid be in range of cellpnum and celllist.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.fillcells_naive_3d!-Tuple{Any, Any, Any}","page":"API","title":"GPUCellListSPH.fillcells_naive_3d!","text":"fillcells_naive_3d!(celllist, cellpnum, pcell)\n\nFill cell list with cell. Naive approach. No bound check. Values in pcell list shoid be in range of cellpnum and celllist.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.makedf-Tuple{SPHProblem}","page":"API","title":"GPUCellListSPH.makedf","text":"makedf(prob::SPHProblem; vtkvars = [\"Density\", \"Acceleration\", \"Velocity\"])\n\nMake DataFrame from SPH Problem.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.neib_external_2d!-NTuple{7, Any}","page":"API","title":"GPUCellListSPH.neib_external_2d!","text":"neib_external_2d!(pairs, cnt, cellpnum, points, celllist, offset, dist)\n\nFind all pairs with another cell shifted on offset.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.neib_external_3d!-NTuple{7, Any}","page":"API","title":"GPUCellListSPH.neib_external_3d!","text":"neib_external_3d!(pairs, cnt, cellpnum, points, celllist, offset, dist)\n\nFind all pairs with another cell shifted on offset.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.neib_internal_2d!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.neib_internal_2d!","text":"neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)\n\nFind all pairs with distance < h in one cell.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.neib_internal_3d!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.neib_internal_3d!","text":"neib_internal_2d!(pairs, cnt, cellpnum, points, celllist, dist)\n\nFind all pairs with distance < h in one cell.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.neighborlist-Tuple{GPUCellList}","page":"API","title":"GPUCellListSPH.neighborlist","text":"neighborlist(c::GPUCellList)\n\nList of pairs with distance.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.partialupdate!-Tuple{GPUCellList}","page":"API","title":"GPUCellListSPH.partialupdate!","text":"partialupdate!(c::GPUCellList)\n\nUpdate only distance \n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.pressure!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.pressure!","text":"pressure!(P, ρ, ρ₀, c₀, γ)\n\nEquation of State in Weakly-Compressible SPH.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.stepsolve!","page":"API","title":"GPUCellListSPH.stepsolve!","text":"stepsolve!(prob::SPHProblem, n::Int = 1; timecall = nothing, timestepping = false, timelims = (-Inf, +Inf))\n\nMake n itarations. \n\ntimestepping - call Δt_stepping for adjust Δt\n\ntimelims - minimal and maximum values for Δt\n\n\n\n\n\n","category":"function"},{"location":"api/#GPUCellListSPH.timesolve!-Tuple{SPHProblem}","page":"API","title":"GPUCellListSPH.timesolve!","text":"timesolve!(prob::SPHProblem; batch = 10, \ntimeframe = 1.0, \nwritetime = 0, \npath = nothing, \npvc = false, \nvtkvars = [\"Acceleration\", \"Velocity\", \"Pressure\"],\ntimestepping = false, \ntimelims = (-Inf, +Inf), \nanim = false,\nplotsettings = Dict(:leg => false))\n\nMake simulation by batch iterations within timeframe. \n\nwritetime - time interval for write vtk / animation.\n\npath - path to export directory.\n\npvc - make PVD file.\n\nvtkvars - variables for export, full list:  [\"Acceleration\", \"Velocity\", \"Pressure\", \"Density\", \"∑W\", \"∑∇W\", \"DPC\"] \n\nanim - make animation.\n\nshowframe - show animation each frame.\n\nplotsettings - keywords for plotting.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update!-Tuple{GPUCellList}","page":"API","title":"GPUCellListSPH.update!","text":"update!(c::GPUCellList)\n\nFull update cell grid.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_all!-NTuple{12, Any}","page":"API","title":"GPUCellListSPH.update_all!","text":"update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_all_3d!-NTuple{13, Any}","page":"API","title":"GPUCellListSPH.update_all_3d!","text":"update_all!(ρ, ρΔt½, v, vΔt½, x, xΔt½, ∑∂ρ∂t, ∑∂v∂t,  Δt, ρ₀, isboundary, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_dpcreg!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.update_dpcreg!","text":"update_dpcreg!(v, x, ∑Δvdpc, Δt, isboundary)\n\nUpdate velocity and position.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_dpcreg_3d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.update_dpcreg_3d!","text":"update_dpcreg!(v, x, ∑Δvdpc, Δt, isboundary)\n\nUpdate velocity and position.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_vp∂v∂tΔt!-NTuple{4, Any}","page":"API","title":"GPUCellListSPH.update_vp∂v∂tΔt!","text":"update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_vp∂v∂tΔt_3d!-NTuple{4, Any}","page":"API","title":"GPUCellListSPH.update_vp∂v∂tΔt_3d!","text":"update_vp∂v∂tΔt!(v, ∑∂v∂t, Δt, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_xpvΔt!-Tuple{Any, Any, Any}","page":"API","title":"GPUCellListSPH.update_xpvΔt!","text":"update_xpvΔt!(x, v, Δt, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_xpvΔt_3d!-NTuple{4, Any}","page":"API","title":"GPUCellListSPH.update_xpvΔt_3d!","text":"update_xpvΔt!(x, v, Δt, ml)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_Δt!-NTuple{9, Any}","page":"API","title":"GPUCellListSPH.update_Δt!","text":"update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary)\n\nUpdate ρ, v, x at timestep Δt and derivatives: ∑∂ρ∂t, ∑∂v∂t.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_ρ!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.update_ρ!","text":"update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.update_ρ_3d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.update_ρ_3d!","text":"update_ρ!(ρ, ∑∂ρ∂t, Δt, ρ₀, isboundary)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.writecsv-Tuple{SPHProblem, Any}","page":"API","title":"GPUCellListSPH.writecsv","text":"writecsv(prob::SPHProblem, path; vtkvars = [\"Density\", \"Acceleration\", \"Velocity\"])\n\nwrite CSV file.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.xsphcorr!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.xsphcorr!","text":"xsphcorr!(∑Δvxsph, v, ρ, W, pairs, m₀)\n\nThe XSPH correction.\n\n\nhattextbfv_i = - epsilon sum m_j fractextbfv_ijoverlinerho_ij W_ij\n\n\nMonaghan JJ (1989) On the problem of penetration in particle methods. J Comput Phys. https://doi.org/10.1016/0021-9991(89)90032-6\n\nCarlos Alberto Dutra Fraga Filho, Reflective Boundary Conditions Coupled With the SPH Method for the Three-Dimensional Simulation of Fluid-Structure Interaction With Solid Boundaries\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.Δt_stepping-NTuple{8, Any}","page":"API","title":"GPUCellListSPH.Δt_stepping","text":"Δt_stepping(buf, a, v, points, c₀, h, CFL, timelims)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.Δt_stepping_3d-NTuple{8, Any}","page":"API","title":"GPUCellListSPH.Δt_stepping_3d","text":"Δt_stepping(buf, a, v, points, c₀, h, CFL, timelims)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.мaxpairs_2d-Tuple{Any}","page":"API","title":"GPUCellListSPH.мaxpairs_2d","text":"мaxpairs_2d(cellpnum)\n\nMaximum number of pairs.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.мaxpairs_3d-Tuple{Any}","page":"API","title":"GPUCellListSPH.мaxpairs_3d","text":"мaxpairs_3d(cellpnum)\n\nMaximum number of pairs.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂v∂t!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.∂v∂t!","text":"∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀)\n\nThe momentum equation (without dissipation).\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂v∂t_3d!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.∂v∂t_3d!","text":"∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀)\n\nThe momentum equation (without dissipation).\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂v∂tpF!-NTuple{7, Any}","page":"API","title":"GPUCellListSPH.∂v∂tpF!","text":"∂v∂tpF!(∑∂v∂t, pairs, points, s, H)\n\nAdd surface tension to ∑∂v∂t. Modified.\n\nA. Tartakovsky and P. Meakin, Phys. Rev. E 72 (2005)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂v∂tpF_3d!-NTuple{7, Any}","page":"API","title":"GPUCellListSPH.∂v∂tpF_3d!","text":"∂v∂tpF!(∑∂v∂t, pairs, points, s, H)\n\nAdd surface tension to ∑∂v∂t. Modified.\n\nA. Tartakovsky and P. Meakin, Phys. Rev. E 72 (2005)\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂Π∂t!-NTuple{10, Any}","page":"API","title":"GPUCellListSPH.∂Π∂t!","text":"∂Π∂t!(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀)\n\nCompute ∂Π∂t - artificial viscosity. Add to ∑∂Π∂t\n\n\nPi_ij = begincases frac- alpha overlinec_ij mu_ij + beta mu_ij^2 overline\rho_ij   textbfv_ijcdot textbfr_ij  0  0   otherwise endcases\n\n\n\n\nmu_ij = frach textbfv_ijcdot textbfr_ijr_ij^2 + eta^2\n\n\n\n\noverlinec_ij  = fracc_i + c_j2\n\n\n\n\noverlinerho_ij = fracrho_i + rho_j2\n\n\n\n\nbeta = 0\n\nc_ij = c_0\n\nm_i = m_j = m_0\n\n\nArtificial viscosity part of momentum equation. \n\n\nfracpartial textbfv_ipartial t = - sum  m_j Pi_ij nabla_i W_ij\n\nJ. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂Π∂t_3d!-NTuple{10, Any}","page":"API","title":"GPUCellListSPH.∂Π∂t_3d!","text":"∂Π∂t!(∑∂Π∂t, ∇Wₙ, pairs, points, h, ρ, α, v, c₀, m₀)\n\nCompute ∂Π∂t - artificial viscosity.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂ρ∂tDDT!-NTuple{14, Any}","page":"API","title":"GPUCellListSPH.∂ρ∂tDDT!","text":"∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype)\n\nCompute ∂ρ∂t - density derivative includind density diffusion.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∂ρ∂tDDT_3d!-NTuple{14, Any}","page":"API","title":"GPUCellListSPH.∂ρ∂tDDT_3d!","text":"∂ρ∂tDDT!(∑∂ρ∂t,  ∇Wₙ, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, MotionLimiter)\n\nCompute ∂ρ∂t - density derivative includind density diffusion.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∇W_2d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.∇W_2d!","text":"∇W_2d!(∇W, pairs, points, kernel, H⁻¹)\n\nCompute gradients. Update ∇W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑W_2d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.∑W_2d!","text":"∑W_2d!(∑W, pairs, sphkernel, H⁻¹)\n\nCompute sum of kernel values for each particles pair in list. Add to ∑W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑W_3d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.∑W_3d!","text":"∑W_2d!(sumW, pairs, sphkernel, H⁻¹)\n\nCompute ∑W for each particles pair in list.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑W∑∇W_2d!-NTuple{7, Any}","page":"API","title":"GPUCellListSPH.∑W∑∇W_2d!","text":"∑W∑∇W_2d!(∑W, ∑∇W, ∇Wₙ, pairs, points, sphkernel, H⁻¹)\n\nCompute ∑W ans gradient for each particles pair in list.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑∇W_2d!-NTuple{5, Any}","page":"API","title":"GPUCellListSPH.∑∇W_2d!","text":"∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹)\n\nCompute gradients. Add sum to ∑∇W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑∇W_2d!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.∑∇W_2d!","text":"∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹)\n\nCompute gradients. Add sum to ∑∇W and update ∇W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"method"},{"location":"api/#GPUCellListSPH.∑∇W_3d!-NTuple{6, Any}","page":"API","title":"GPUCellListSPH.∑∇W_3d!","text":"∑∇W_2d!(sum∇W, ∇Wₙ, pairs, points, kernel, H⁻¹)\n\nCompute gradients.\n\n\n\n\n\n","category":"method"},{"location":"details/#Details","page":"Details","title":"Details","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"m_i, m_j","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"m_0","category":"page"},{"location":"details/#Artificial-Viscosity","page":"Details","title":"Artificial Viscosity","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"\nPi_ij = begincases frac- alpha overlinec_ij mu_ij + beta mu_ij^2 overlinerho_ij   textbfv_ijcdot textbfr_ij  0  0   otherwise endcases\n\n\n\noverlinec_ij  = fracc_i + c_j2\n\n\n\noverlinerho_ij = fracrho_i + rho_j2\n","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"Monaghan style artificial viscosity:","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"\nfracpartial textbfv_ipartial t = - sum  m_j Pi_ij nabla_i W_ij","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"J. Monaghan, “Smoothed particle hydrodynamics”, Reports on Progress in Physics, 68 (2005), pp. 1703-1759.","category":"page"},{"location":"details/#Momentum-Equation-with-Artificial-Viscosity","page":"Details","title":"Momentum Equation with Artificial Viscosity","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"fracpartial textbfv_ipartial t = - sum  m_j left( fracb_irho^2_i + fracb_jrho^2_j + Pi_ij right) nabla_i W_ij\n","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"J. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.","category":"page"},{"location":"details/#Continuity-equation","page":"Details","title":"Continuity equation","text":"","category":"section"},{"location":"details/#Density-diffusion-term","page":"Details","title":"Density diffusion term","text":"","category":"section"},{"location":"details/#XSPH-correction","page":"Details","title":"XSPH correction","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"hattextbfv_i = - epsilon sum m_j fractextbfv_ijoverlinerho_ij W_ij","category":"page"},{"location":"details/#Corrected-Smoothed-Particle-Method-(CSPM)","page":"Details","title":"Corrected Smoothed Particle Method (CSPM)","text":"","category":"section"},{"location":"details/#Dynamic-Particle-Collision-(DPC)-correction.","page":"Details","title":"Dynamic Particle Collision (DPC) correction.","text":"","category":"section"},{"location":"details/","page":"Details","title":"Details","text":"delta textbfv_i^DPC = sum k_ijfracm_jm_i + m_jv_ij^coll + fracDelta  trho_isum phi_ij frac2V_jV_i + V_jfracp_ij^br_ij^2 + eta^2textbfr_ij\n\n\n\n(v_ij^coll  quad phi_ij) = begincases (fractextbfv_ijcdot textbfr_ijr_ij^2 + eta^2textbfr_ji quad 0)  textbfv_ijcdot textbfr_ij  0  (0 quad 1)   otherwise endcases\n\n\n\np_ij^b = tildep_ij chi_ij \n\n\n\ntildep_ij = max(min(lambda p_i + p_j lambda p_max) p_min)\n\n\n\nchi_ij  = sqrtfracomega(r_ij l_0)omega(l_02 l_0)\n\n\n\nk_ij =  begincases chi_ij  05 le r_ijl_0  1  1  r_ijl_0  05 endcases\n","category":"page"},{"location":"details/","page":"Details","title":"Details","text":"Mojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf","category":"page"},{"location":"details/#Time-stepping","page":"Details","title":"Time stepping","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GPUCellListSPH","category":"page"},{"location":"#GPUCellListSPH","page":"Home","title":"GPUCellListSPH","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GPUCellListSPH.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Docs","page":"Home","title":"Docs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.GPUCellList","category":"page"},{"location":"#GPUCellListSPH.GPUCellList","page":"Home","title":"GPUCellListSPH.GPUCellList","text":"GPUCellList(points, cellsize, dist; mppcell = 0, mpairs = 0)\n\nMake cell list structure.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.update!","category":"page"},{"location":"#GPUCellListSPH.update!","page":"Home","title":"GPUCellListSPH.update!","text":"update!(c::GPUCellList)\n\nFull update cell grid.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.partialupdate!","category":"page"},{"location":"#GPUCellListSPH.partialupdate!","page":"Home","title":"GPUCellListSPH.partialupdate!","text":"partialupdate!(c::GPUCellList)\n\nUpdate only distance \n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.neighborlist","category":"page"},{"location":"#GPUCellListSPH.neighborlist","page":"Home","title":"GPUCellListSPH.neighborlist","text":"neighborlist(c::GPUCellList)\n\nList of pairs with distance.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∑W_2d!","category":"page"},{"location":"#GPUCellListSPH.∑W_2d!","page":"Home","title":"GPUCellListSPH.∑W_2d!","text":"∑W_2d!(∑W, pairs, sphkernel, H⁻¹)\n\nCompute sum of kernel values for each particles pair in list. Add to ∑W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∑∇W_2d!","category":"page"},{"location":"#GPUCellListSPH.∑∇W_2d!","page":"Home","title":"GPUCellListSPH.∑∇W_2d!","text":"∑∇W_2d!(∑∇W, ∇W, pairs, points, kernel, H⁻¹)\n\nCompute gradients. Add sum to ∑∇W and update ∇W. See SPHKernels.jl for details.\n\n\n\n\n\n∑∇W_2d!(∑∇W, pairs, points, kernel, H⁻¹)\n\nCompute gradients. Add sum to ∑∇W. See SPHKernels.jl for details.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∂ρ∂tDDT!","category":"page"},{"location":"#GPUCellListSPH.∂ρ∂tDDT!","page":"Home","title":"GPUCellListSPH.∂ρ∂tDDT!","text":"∂ρ∂tDDT!(∑∂ρ∂t,  ∇W, pairs, points, h, m₀, δᵩ, c₀, γ, g, ρ₀, ρ, v, ptype)\n\nCompute ∂ρ∂t - density derivative includind density diffusion.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∂Π∂t!","category":"page"},{"location":"#GPUCellListSPH.∂Π∂t!","page":"Home","title":"GPUCellListSPH.∂Π∂t!","text":"∂Π∂t!(∑∂Π∂t, ∇W, pairs, points, h, ρ, α, v, c₀, m₀)\n\nCompute ∂Π∂t - artificial viscosity. Add to ∑∂Π∂t\n\n\nPi_ij = begincases frac- alpha overlinec_ij mu_ij + beta mu_ij^2 overline\rho_ij   textbfv_ijcdot textbfr_ij  0  0   otherwise endcases\n\n\n\n\nmu_ij = frach textbfv_ijcdot textbfr_ijr_ij^2 + eta^2\n\n\n\n\noverlinec_ij  = fracc_i + c_j2\n\n\n\n\noverlinerho_ij = fracrho_i + rho_j2\n\n\n\n\nbeta = 0\n\nc_ij = c_0\n\nm_i = m_j = m_0\n\n\nArtificial viscosity part of momentum equation. \n\n\nfracpartial textbfv_ipartial t = - sum  m_j Pi_ij nabla_i W_ij\n\nJ. Monaghan, Smoothed Particle Hydrodynamics, “Annual Review of Astronomy and Astrophysics”, 30 (1992), pp. 543-574.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∂v∂t!","category":"page"},{"location":"#GPUCellListSPH.∂v∂t!","page":"Home","title":"GPUCellListSPH.∂v∂t!","text":"∂v∂t!(∑∂v∂t,  ∇Wₙ, pairs, m, ρ, c₀, γ, ρ₀)\n\nThe momentum equation (without dissipation).\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.pressure","category":"page"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.∂v∂tpF!","category":"page"},{"location":"#GPUCellListSPH.∂v∂tpF!","page":"Home","title":"GPUCellListSPH.∂v∂tpF!","text":"∂v∂tpF!(∑∂v∂t, pairs, points, s, H)\n\nAdd surface tension to ∑∂v∂t. Modified.\n\nA. Tartakovsky and P. Meakin, Phys. Rev. E 72 (2005)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.dpcreg!","category":"page"},{"location":"#GPUCellListSPH.dpcreg!","page":"Home","title":"GPUCellListSPH.dpcreg!","text":"dpcreg!(∑Δvdpc, v, ρ, P, pairs, points, sphkernel, l₀, Pmin, Pmax, Δt, λ, dpckernlim)\n\nDynamic Particle Collision (DPC) correction.\n\ndelta textbfv_i^DPC = sum k_ijfracm_jm_i + m_jv_ij^coll + fracDelta  trho_isum phi_ij frac2V_jV_i + V_jfracp_ij^br_ij^2 + eta^2textbfr_ij\n\n\n\n(v_ij^coll  quad phi_ij) = begincases (fractextbfv_ijcdot textbfr_ijr_ij^2 + eta^2textbfr_ji quad 0)  textbfv_ijcdot textbfr_ij  0  (0 quad 1)   otherwise endcases\n\n\np_ij^b = tildep_ij chi_ij \n\n\n\ntildep_ij = max(min(lambda p_i + p_j lambda p_max) p_min)\n\n\n\nchi_ij  = sqrtfracomega(r_ij l_0)omega(l_02 l_0)\n\n\n\nk_ij =  begincases chi_ij  05 le r_ijl_0  1  1  r_ijl_0  05 endcases\n\n\nMojtaba Jandaghian, Herman Musumari Siaben, Ahmad Shakibaeinia, Stability and accuracy of the weakly compressible SPH with particle regularization techniques https://arxiv.org/pdf/2110.10076.pdf\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.makedf","category":"page"},{"location":"#GPUCellListSPH.makedf","page":"Home","title":"GPUCellListSPH.makedf","text":"makedf(prob::SPHProblem; vtkvars = [\"Density\", \"Acceleration\", \"Velocity\"])\n\nMake DataFrame from SPH Problem.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"GPUCellListSPH.writecsv","category":"page"},{"location":"#GPUCellListSPH.writecsv","page":"Home","title":"GPUCellListSPH.writecsv","text":"writecsv(prob::SPHProblem, path; vtkvars = [\"Density\", \"Acceleration\", \"Velocity\"])\n\nwrite CSV file.\n\n\n\n\n\n","category":"function"}]
}
