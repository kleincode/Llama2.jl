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
                @test tokenizer.token_to_index[tokens[i]] == i
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
        # Load Andrew Karpathy's tokenizer from the llama2.c repository
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

    @testset "Decode" begin
        tokens = ["ERR", "<bos>", "<eos>", "a", "b", "c", "<0x2C>", " dog"]
        scores = ones(Float32, size(tokens))
        tokenizer = Tokenizer(tokens, scores)
        @testset "Decode returns string" begin
            return_types = Base.return_types(decode)
            @test length(return_types) == 1
            @test return_types[1] == String
        end
        @testset "Decode regular token" begin
            @test decode(tokenizer, 1, 3) == "a"
            @test decode(tokenizer, 3, 4) == "b"
            @test decode(tokenizer, 6, 5) == "c"
            @test decode(tokenizer, 6, 7) == " dog"
        end
        @testset "Decode control chars like <bos> and <eos>" begin
            @test decode(tokenizer, 3, 1) == "<bos>"
            @test decode(tokenizer, 1, 2) == "<eos>"
        end
        @testset "Decode char of form <0xhh>" begin
            @test decode(tokenizer, 1, 6) == ","
        end
        @testset "Decode with previous token == BOS" begin
            @test decode(tokenizer, 1, 2) == "<eos>"
            @test decode(tokenizer, 1, 4) == "b"
            @test decode(tokenizer, 1, 6) == ","
            @test decode(tokenizer, 1, 7) == "dog" # space is stripped
        end
        @testset "Out of bounds token" begin
            @test_throws BoundsError decode(tokenizer, 1, length(tokens))
            @test_throws BoundsError decode(tokenizer, 1, -1)
        end
    end

    @testset "Encode" begin
        @testset "Simple Tokenizer" begin
            simple_tokens = ["<bos>", "<eos>", "a", "b", "c", " "]
            simple_scores = ones(Float32, size(simple_tokens))
            simple_tokenizer = Tokenizer(simple_tokens, simple_scores)

            text = " aacb"
            encoded_tokens = encode(simple_tokenizer, text)
            @test encoded_tokens == [6, 3, 3, 5, 4]
        end

        """
        vocab_size = 32000
        tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", vocab_size)
        @testset "Check different strings" begin
            texts = ["Good morning", "1234", "hello!", "Good morning", "Good Morning", "abcdef", "1"]
            for text in texts
                decoded_text = ""
                tokens = encode(tokenizer, text)
                for i in 1:(length(tokens)-1)
                    piece = decode(tokenizer, tokens[i], tokens[i+1])
                    decoded_text = decoded_text * piece
                end
                
                @test decoded_text == text
            end
        end 

        @testset "Letters and Numbers" begin
            texts = ["a", "b", "c", "d", "1", "0", "2", "?", "!"]
            for text in texts
                tokens = encode(tokenizer, text)
                decoded_text = ""
                for i in 1:(length(tokens)-1)
                    @test tokens[i] == "g"
                    piece = decode(tokenizer, tokens[i], tokens[i+1])
                    decoded_text = decoded_text * piece
                end
                
                @test decoded_text == text
            end
        end
        @testset "Empty string" begin
            token = encode(tokenizer, "")
            @test token == []
        end
        """

        
    end
end