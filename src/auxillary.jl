

function loadparticles(fluid_csv, boundary_csv; fluidcols = ["Points:0", "Points:2"], boundarycols = ["Points:0", "Points:2"])
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    points = Tuple.(eachrow(DF_FLUID[!, fluidcols]))
    append!(points, Tuple.(eachrow(DF_BOUND[!, boundarycols])))

    return points, DF_FLUID, DF_BOUND
end

"""
    makedf(prob::SPHProblem; vtkvars = ["Density", "Acceleration", "Velocity"])

Make DataFrame from SPH Problem.
"""
function makedf(prob::SPHProblem; vtkvars = ["Density", "Acceleration", "Velocity"])

    cpupoints    = Array(get_points(prob))
    dim          = length(first(cpupoints))
    dfn          = ["x", "y", "z"]
    dfarr        = [dfn[i] => map(x -> x[i], cpupoints) for i in 1:dim]

    if "Density"      in vtkvars push!(dfarr, "Density"  => Array(get_density(prob))) end
    if "Pressure"     in vtkvars push!(dfarr, "Pressure" => Array(get_pressure(prob))) end
    if "Acceleration" in vtkvars 
        arrs = Array.(get_acceleration(prob))
        for i in eachindex(arrs)
            push!(dfarr, "Acceleration_"*dfn[i] => arrs[i])
        end
    end

    if "Velocity" in vtkvars 
            av           = Array(get_velocity(prob))
            varr        = ["Velocity_"*dfn[i] => map(x -> x[i], av) for i in 1:dim]
            append!(dfarr, varr)        
    end

    if "∑W" in vtkvars push!(dfarr, "∑W" => Array(get_sumw(prob))) end



    if "∑∇W" in vtkvars 
        arrs = Array.(get_sumgradw(prob)) 
        for i in eachindex(arrs)
            push!(dfarr, "∑∇W_"*dfn[i] => arrs[i])
        end
    end
        
    if "DPC" in vtkvars 
        arrs = Array.(get_dpccorr(prob)) 
        for i in eachindex(arrs)
            push!(dfarr, "DPC_"*dfn[i] => arrs[i])
        end
    end

        DataFrame(dfarr)
end

"""
    writecsv(prob::SPHProblem, path; vtkvars = ["Density", "Acceleration", "Velocity"])

write CSV file.
"""
function writecsv(prob::SPHProblem, path; vtkvars = ["Density", "Acceleration", "Velocity"])
    dirn = dirname(path)
    if !isdir(dirn) error("Wrong path.") end 
    CSV.write(path, makedf(prob; vtkvars = vtkvars))
end
