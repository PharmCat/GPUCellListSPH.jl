mutable struct SPHProblem
    dim
    h
    h⁻¹
    H
    H⁻¹
    kernel
    ∑W
    ∑∇W
    ∇Wₙ
    ∑∂Π∂t
    ∑∂v∂t
    ∑∂ρ∂t
    function SPHProblem(dim, h, H, kernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t)
        new{}(dim, h, 1/h, H, 1/H, kernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t)
    end
end



function solve!(prob::SPHProblem, system::GPUCellList)

end

