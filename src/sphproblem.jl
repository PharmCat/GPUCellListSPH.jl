mutable struct SPHProblem
    h
    h⁻¹
    H
    kernel
    sumW

    function SPHProblem(h, H, kernel, sumW)
        new{}(h, 1/h⁻¹, H, kernel, sumW)
    end
end



function solve!(prob::SPHProblem, system::GPUCellList)

end

