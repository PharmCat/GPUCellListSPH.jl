

function loadparticles(fluid_csv, boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    points = Tuple.(eachrow(DF_FLUID[!, ["Points:0", "Points:2"]]))
    append!(points, Tuple.(eachrow(DF_BOUND[!, ["Points:0", "Points:2"]])))

    return points, DF_FLUID, DF_BOUND
end
