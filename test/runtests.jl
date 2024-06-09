using Llama2
using Test

@testset "Llama2.jl" begin
    include("test_tokenizer.jl")
    include("test_sampler.jl")
    include("test_config.jl")
    include("test_transformer.jl")
    include("test_math_llama.jl")
end
