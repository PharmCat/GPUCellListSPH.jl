function create_vtp_file(filename, x, expdict; pvd = nothing, time = nothing)
    polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
    verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

    vtk_grid(filename, x..., polys, verts, compress = true, append = false) do vtk
        for (k, v) in expdict
            vtk[k] = v
        end
        if !isnothing(pvd) && !isnothing(time) 
            pvd[time] = vtk 
        end
    end
end

"""
    writevtk(prob::SPHProblem, filename, vtkvars, cpupoints = nothing; pvd = nothing, writetime = false)

Create vtp file.

`prob` - SPH problem;

`filename` - path to file;

`vtkvars` - list of variables: `["Density", "Pressure", "Type", "Acceleration", "Velocity", "∑W", "∑∇W", "DPC"]`;

`cpupoints` - coordinates in CPU memory (optional).
"""
function writevtk(prob::SPHProblem, filename, vtkvars, cpupoints = nothing; pvd = nothing, writetime = false)
    expdict                 = Dict()
    if isnothing(cpupoints) cpupoints = Array.(get_points(prob)) end
    coordsarr               = collect(cpupoints)
    if "Density"      in vtkvars expdict["Density"]      = Array(get_density(prob)) end
    if "Pressure"     in vtkvars expdict["Pressure"]     = Array(get_pressure(prob)) end
    if "Type"         in vtkvars expdict["Type"]          = Array(get_ptype(prob)) end
    if "Acceleration" in vtkvars expdict["Acceleration"] = Array.(get_acceleration(prob)) end
    if "Velocity"     in vtkvars  expdict["Velocity"]     = Array.(get_velocity(prob)) end
    if "∑W" in vtkvars expdict["∑W"]           = Array(get_sumw(prob)) end
    if "∑∇W" in vtkvars expdict["∑∇W"]         = Array.(get_sumgradw(prob)) end
    if "DPC" in vtkvars expdict["DPC"]         = Array.(get_dpccorr(prob)) end

    create_vtp_file(filename, coordsarr, expdict; pvd = pvd, time = ifelse(writetime, prob.etime, nothing))
end


function readvtk(prob::SPHProblem)

end

