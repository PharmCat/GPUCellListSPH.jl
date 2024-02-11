

function create_vtp_file(filename, x, ρ, ∑∂v∂t, v)
    # Convert the particle positions and densities into the format required by the vtk_grid function:
    ax = Array(x)
    points = zeros(Float64, 2, length(ax))
    for (i, r) in enumerate(eachcol(points))
        r .= ax[i]
    end

    polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
    verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

    # Note: the order of verts, lines, polys and strips is not important.
    # One doesn't even need to pass all of them.
    all_cells  = (verts, polys)

    varr        = Array(v)
    v           = zeros(Float64, 2, length(varr))
    for (i, r) in enumerate(eachcol(v))
        r .= varr[i]
    end


    # Create a .vtp file with the particle positions and densities:
    vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk
        # Add the particle densities as a point data array:
        vtk_point_data(vtk, Array(ρ), "Density")
        vtk_point_data(vtk, permutedims(Array(∑∂v∂t)), "Acceleration")
        vtk_point_data(vtk, v, "Velocity")
    end
end


# Initialize containers for VTK data structure
#=
polys = empty(MeshCell{WriteVTK.PolyData.Polys, UnitRange{Int64}}[])
verts = empty(MeshCell{WriteVTK.PolyData.Verts, UnitRange{Int64}}[])
all_cells = (verts, polys)

save_points  = [SVector(t[1], t[2], 0.0) for t in Array(get_points(sphprob))]

vtk_grid(raw"E:\SPH\TestOfFileWriteVTK.vtp", save_points, all_cells...) do vtk
    vtk_point_data(vtk, Array(get_density(sphprob)), "Density")
end;
=#