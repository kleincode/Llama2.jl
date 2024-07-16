#=
If you get a LoadError like "Package Llama2 is required but does not seem to be installed",
check out https://discourse.julialang.org/t/documenter-jl-expects-my-local-project-to-be-registered/101437/2
This solved it for me.
=#
using Llama2
using Documenter

DocMeta.setdocmeta!(Llama2, :DocTestSetup, :(using Llama2); recursive=true)

makedocs(;
    modules=[Llama2],
    authors="Thomas Fischer <t.fischer.1@campus.tu-berlin.de>, Johanna Giese <j.giese@campus.tu-berlin.de>, Janik Häußer <janik.haeusser@campus.tu-berlin.de>, Felix Kleinsteuber <f.kleinsteuber@campus.tu-berlin.de>",
    sitename="Llama2.jl",
    format=Documenter.HTML(;
        canonical="https://kleincode.github.io/Llama2.jl", edit_link="main", assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "example.md",
        "Performance" => "performance.md",
    ],
)

deploydocs(; repo="github.com/kleincode/Llama2.jl", devbranch="main")
