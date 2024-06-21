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

    @testset "Read tokenizer.bin from llama2.c repository" begin
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
            @test decode(tokenizer, 2, 4) == "a"
            @test decode(tokenizer, 4, 5) == "b"
            @test decode(tokenizer, 7, 6) == "c"  
            @test decode(tokenizer, 7, 8) == " dog"
        end
        @testset "Decode control chars like <bos> and <eos>" begin
            @test decode(tokenizer, 4, 2) == "<bos>"
            @test decode(tokenizer, 2, 3) == "<eos>"
        end
        @testset "Decode char of form <0xhh>" begin
            @test decode(tokenizer, 2, 7) == ","
        end
        @testset "Decode with previous token == BOS" begin
            @test decode(tokenizer, 2, 3) == "<eos>"
            @test decode(tokenizer, 2, 5) == "b"
            @test decode(tokenizer, 2, 7) == ","
            @test decode(tokenizer, 2, 8) == "dog" # space is stripped
        end
        @testset "Out of bounds token" begin
            @test_throws BoundsError decode(tokenizer, 2, length(tokens) + 1)
            @test_throws BoundsError decode(tokenizer, 2, -1)
        end
    end

    @testset "Encode" begin
        @testset "Simple Tokenizer" begin
            simple_tokens = [" ", "<bos>", "<eos>", "a", "b", "c"]
            simple_scores = ones(Float32, size(simple_tokens))
            simple_tokenizer = Tokenizer(simple_tokens, simple_scores)

            text = "aacb"   
            encoded_tokens = encode(simple_tokenizer, text)
            @test encoded_tokens == [2, 1, 4, 4, 6, 5]

            @test decode(simple_tokenizer, 1, 4) == "a"
            @test decode(simple_tokenizer, 4, 4) == "a"
            @test decode(simple_tokenizer, 4, 6) == "c"
            @test decode(simple_tokenizer, 6, 5) == "b"
            
            simple_decoded_text = ""
            for i in 1:(length(encoded_tokens)-1)
                piece = decode(simple_tokenizer, encoded_tokens[i], encoded_tokens[i+1])
                simple_decoded_text = simple_decoded_text * piece
            end

            @test simple_decoded_text == "aacb" # EOS token added to string
        end

        
        vocab_size = 32000
        tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", vocab_size)
        
        @testset "Empty string" begin
            token = encode(tokenizer, "")
            @test token == [2]    # BOS token
        end

        @testset "Letters and Numbers" begin
            texts = ["a", "b", "c", "d", "!", ",", "2", "<0x2C>"]   
            for text in texts
                tokens = encode(tokenizer, text)
                decoded_text = ""
                for i in 1:(length(tokens)-1)
                    piece = decode(tokenizer, tokens[i], tokens[i+1])
                    decoded_text = decoded_text * piece
                end
                
                @test decoded_text == text
            end
        end

        @testset "Check different strings" begin
            texts = ["Good morning", "1234", "hello!", "Good morning", "abcdef"]
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

        @testset "Check EOS token" begin
            texts = ["Good morning", "1234", "hello!", "Good morning", "abcdef"]
            for text in texts
                decoded_text = ""
                tokens = encode(tokenizer, text, true)
                for i in 1:(length(tokens)-1)
                    piece = decode(tokenizer, tokens[i], tokens[i+1])
                    decoded_text = decoded_text * piece
                end
                
                @test decoded_text == text * "\n</s>\n"
            end 
        end
        
    end
end