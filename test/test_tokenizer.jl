using Llama2
using Test

@testset "Tokenizer" begin
    @testset "Construct from list of tokens and scores" begin
        @testset "Good case" begin
            tokens = ["a", "b", "c"]
            scores = [1.0f0, 2.0f0, 3.0f0]
            tokenizer = Tokenizer(tokens, scores)

            @test length(tokenizer.index_to_token) == 3
            @test length(tokenizer.token_to_index) == 3
            @test length(tokenizer.vocab_scores) == 3

            for i in 1:3
                @test tokenizer.index_to_token[i] == tokens[i]
                @test tokenizer.vocab_scores[i] == scores[i]
            end
        end
        @testset "Empty token list" begin
            @test_throws ArgumentError Tokenizer(Vector{String}([]), Vector{Float32}([]))
        end
        @testset "Mismatched lengths" begin
            @test_throws ArgumentError Tokenizer(["a"], Vector{Float32}([]))
            @test_throws ArgumentError Tokenizer(Vector{String}([]), [1.0f0])
            @test_throws ArgumentError Tokenizer(["a"], [1.0f0, 2.0f0])
            @test_throws ArgumentError Tokenizer(["a", "b"], [1.0f0])
        end
        @testset "A token is the empty string" begin
            @test_throws ArgumentError Tokenizer([""], [1.0f0])
            @test_throws ArgumentError Tokenizer(["a", ""], [1.0f0, 2.0f0])
        end
        @testset "Duplicate tokens" begin
            @test_throws ArgumentError Tokenizer(["a", "a"], [1.0f0, 2.0f0])
        end
    end

    @testset "Read tokenizer.bin from llama.c repository" begin
        # Load Andrew Karpathy's tokenizer from the llama.c repository
        vocab_size = 32000
        tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", vocab_size)

        # All vectors must have length vocab_size
        @test length(tokenizer.index_to_token) == vocab_size
        @test length(tokenizer.token_to_index) == vocab_size
        @test length(tokenizer.vocab_scores) == vocab_size

        for i in 1:vocab_size
            # token_to_index should invert index_to_token
            @test tokenizer.token_to_index[tokenizer.index_to_token[i]] == i
            # The only thing we know about the scores is that they should be finite
            @test isfinite(tokenizer.vocab_scores[i])
        end
    end
end