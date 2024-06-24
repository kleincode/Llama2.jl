@testset "Read Karpathy" begin
    @testset "stories15M.bin" begin
        llama_file = get_stories15M()
        config, weights = read_karpathy(llama_file)
        @test typeof(config) == Config
        @test typeof(weights) == TransformerWeights
        @test weights.wcls == weights.token_embedding_table
    end
end