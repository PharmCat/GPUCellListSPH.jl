

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
    v          = zeros(Float64, length(varr), 2)
    for (i, r) in enumerate(eachrow(v))
        r .= varr[i]
    end


    # Create a .vtp file with the particle positions and densities:
    vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk
        # Add the particle densities as a point data array:
        vtk_point_data(vtk, Array(ρ), "Density")
        vtk_point_data(vtk, Array(∑∂v∂t), "Acceleration")
        vtk_point_data(vtk, v, "Velocity")
    end
end