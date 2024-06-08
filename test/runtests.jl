using Llama2
using Test

@testset "Llama2.jl" begin
    include("test_tokenizer.jl")
    include("test_config.jl")
    include("test_transformer.jl")
end
