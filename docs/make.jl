using Llama2
using Documenter

DocMeta.setdocmeta!(Llama2, :DocTestSetup, :(using Llama2); recursive=true)

makedocs(;
    modules=[Llama2],
    authors="Thomas Fischer <t.fischer.1@campus.tu-berlin.de>, Johanna Giese <j.giese@campus.tu-berlin.de>, Janik Häußer <janik.haeusser@campus.tu-berlin.de>, Felix Kleinsteuber <f.kleinsteuber@campus.tu-berlin.de>",
    sitename="Llama2.jl",
    format=Documenter.HTML(;
        canonical="https://kleincode.github.io/Llama2.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kleincode/Llama2.jl",
    devbranch="main",
)
