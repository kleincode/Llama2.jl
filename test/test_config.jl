using Llama2
using Test

@testset "Config" begin
    @testset "Initialize configuration" begin
        @testset "Good case" begin
            # initialization should not throw an error
            dim::Int32 = 16
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 8
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        end

        @testset "Dimension is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 0
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Hidden dimension is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 0
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Number of layers is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 0
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Number of query heads is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 0
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Number of key/value heads is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 0
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Vocabulary is empty" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 0
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "Sequence length is <= 0" begin
            # initialization should throw an error
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 0

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "dim is not a multiple of n_heads" begin
            dim::Int32 = 7
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 4
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end

        @testset "n_heads is not a multiple of n_kv_heads" begin
            dim::Int32 = 8
            hidden_dim::Int32 = 10
            n_layers::Int32 = 3
            n_heads::Int32 = 2
            n_kv_heads::Int32 = 4
            vocab_size::Int32 = 30
            seq_len::Int32 = 2

            @test_throws ArgumentError Config(
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
            )
        end
    end
end