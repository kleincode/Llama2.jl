using Llama2
using Test
using Downloads

@testset "Transformer" begin
    @testset "Initialize Transformer Weights" begin
        # initialize Config
        dim::Int32 = 8
        hidden_dim::Int32 = 10
        n_layers::Int32 = 3
        n_heads::Int32 = 4
        n_kv_heads::Int32 = 4
        vocab_size::Int32 = 30
        seq_len::Int32 = 2

        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        weights = TransformerWeights(config)

        head_size::Int32 = dim ÷ n_heads

        # test size of Transformer Weights
        @test size(weights.token_embedding_table) == (dim, vocab_size)
        @test size(weights.rms_att_weight) == (dim, n_layers)
        @test size(weights.rms_ffn_weight) == (dim, n_layers)

        @test size(weights.wq) == ((n_heads * head_size), dim, n_layers)
        @test size(weights.wk) == (dim, (n_kv_heads * head_size), n_layers)
        @test size(weights.wv) == (dim, (n_kv_heads * head_size), n_layers)
        @test size(weights.wo) == (dim, (n_heads * head_size), n_layers)

        @test size(weights.w1) == (dim, hidden_dim, n_layers)
        @test size(weights.w2) == (hidden_dim, dim, n_layers)
        @test size(weights.w3) == (dim, hidden_dim, n_layers)

        @test size(weights.rms_final_weight) == (dim,)
    end

    @testset "Initialize RunState" begin
        # initialize Config
        dim::Int32 = 8
        hidden_dim::Int32 = 10
        n_layers::Int32 = 3
        n_heads::Int32 = 4
        n_kv_heads::Int32 = 4
        vocab_size::Int32 = 30
        seq_len::Int32 = 2

        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        state = RunState(config)

        kv_dim::Int32 = (dim * n_kv_heads) ÷ n_heads

        # test size of RunState parameters
        @test size(state.x) == (dim,)
        @test size(state.xb) == (dim,)
        @test size(state.xb2) == (dim,)
        @test size(state.hb) == (hidden_dim,)
        @test size(state.hb2) == (hidden_dim,)
        @test size(state.q) == (dim,)
        @test size(state.att) == (n_heads, seq_len)
        @test size(state.logits) == (vocab_size,)
        @test size(state.key_cache) == (n_layers, seq_len, kv_dim)
        @test size(state.value_cache) == (n_layers, seq_len, kv_dim)
    end

    @testset "Transformer forward with dummy weights" begin
        # initialize Config
        dim::Int32 = 8
        hidden_dim::Int32 = 10
        n_layers::Int32 = 3
        n_heads::Int32 = 4
        n_kv_heads::Int32 = 2
        vocab_size::Int32 = 30
        seq_len::Int32 = 10

        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        state = RunState(config)
        weights = TransformerWeights(config)
        transformer = Transformer(config, weights, state, 0, Float32[], 0)

        for i in 1:seq_len
            forward(transformer, i, i)
        end
    end

    @testset "Read model.bin file from Karpathy" begin
        llama_file = "../bin/transformer/stories15M.bin"
        if !isfile(llama_file)
            println("Downloading stories15M.bin...")
            Downloads.download(
                "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
                llama_file,
            )
            println("Download complete!")
        end
        @testset "Read Config from Bin File" begin
            config, _ = open_file(llama_file)
            @test typeof(config) == Config
        end
        @testset "Read TransformerWeights from Bin File" begin
            _, weights = open_file(llama_file)
            @test typeof(weights) == TransformerWeights
        end
    end
end
