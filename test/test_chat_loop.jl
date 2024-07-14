using Llama2
using Test

@testset "Chat Loop" begin
    @testset "try it out" begin
        config, weights = read_karpathy(get_stories15M())
        state = RunState{Float32}(config)
        transformer = Transformer{Float32}(config, weights, state)
        tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", 32000)
        sampler = Sampler{Float32}(1.0f0, 0.9f0, 420)
        chat(transformer, tokenizer, sampler)
    end
end