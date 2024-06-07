using Llama2
using Test

@testset "Transformer" begin
    @testset "Initialize Transformer Weights" begin
        # initialize Config
        dim::Int32 = 5
        hidden_dim::Int32 = 10
        n_layers::Int32 = 3
        n_heads::Int32 = 4
        n_kv_heads::Int32 = 6
        vocab_size::Int32 = 30
        seq_len::Int32 = 2

        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        weights = TransformerWeights(config)

        head_size::Int32 = dim ÷ n_heads

        # test size of Transformer Weights
        @test size(weights.token_embedding_table) == (vocab_size, dim)
        @test size(weights.rms_att_weight) == (n_layers, dim)
        @test size(weights.rms_ffn_weight) == (n_layers, dim)

        @test size(weights.wq) == (n_layers, dim, (n_heads * head_size))
        @test size(weights.wk) == (n_layers, dim, (n_kv_heads * head_size))
        @test size(weights.wv) == (n_layers, dim, (n_kv_heads * head_size))
        @test size(weights.wo) == (n_layers, (n_heads * head_size), dim)

        @test size(weights.w1) == (n_layers, hidden_dim, dim)
        @test size(weights.w2) == (n_layers, dim, hidden_dim)
        @test size(weights.w3) == (n_layers, hidden_dim, dim)

        @test size(weights.rms_final_weight) == (dim,)
    end

    @testset "Initialize RunState" begin
        # initialize Config
        dim::Int32 = 5
        hidden_dim::Int32 = 10
        n_layers::Int32 = 3
        n_heads::Int32 = 4
        n_kv_heads::Int32 = 6
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
        @test size(state.k) == (dim,)
        @test size(state.v) == (dim,)
        @test size(state.att) == (n_heads, seq_len)
        @test size(state.logits) == (vocab_size,)
        @test size(state.key_cache) == (n_layers, seq_len, kv_dim)
        @test size(state.value_cache) == (n_layers, seq_len, kv_dim)
    end
end