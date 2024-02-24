using GPUCellListSPH
using Documenter

DocMeta.setdocmeta!(GPUCellListSPH, :DocTestSetup, :(using GPUCellListSPH); recursive=true)

makedocs(;
    modules=[GPUCellListSPH],
    authors="Vladimir Arnautov",
    repo="https://github.com/PharmCat/GPUCellListSPH.jl/blob/{commit}{path}#{line}",
    sitename="GPUCellListSPH.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PharmCat.github.io/GPUCellListSPH.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Details" => "details.md",
        "API"  => "api.md",
    ],
)

deploydocs(;
    repo="github.com/PharmCat/GPUCellListSPH.jl",
    devbranch="main",
)
