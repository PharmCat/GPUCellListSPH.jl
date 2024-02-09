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
    ρΔt½ 
    vΔt½
    xΔt½
    ml
    gf
    isboundary
    function SPHProblem(dim, h, H, kernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t, ρΔt½, vΔt½, xΔt½, ml, gf, isboundary)
        new{}(dim, h, 1/h, H, 1/H, kernel, ∑W, ∑∇W, ∇Wₙ, ∑∂Π∂t, ∑∂v∂t, ∑∂ρ∂t, ρΔt½, vΔt½, xΔt½, ml, gf, isboundary)
    end
end



function solve!(prob::SPHProblem, system::GPUCellList)

end

