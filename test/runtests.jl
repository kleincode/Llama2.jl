using Llama2
using Test

include("utils.jl")

@testset "Llama2.jl" begin
    include("test_math_llama.jl")
    include("test_tokenizer.jl")
    include("test_sampler.jl")
    include("test_config.jl")
    include("test_read_karpathy.jl")
    include("test_transformer.jl")
    include("test_generate.jl")
end
