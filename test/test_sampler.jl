using Llama2
using Test

@testset "Sampler Tests" begin
    @testset "Sampler constructor" begin
        sampler_default = Sampler()
        @test sampler_default.temperature == Float32(1.0)
        @test sampler_default.topp == Float32(0.0)

        sampler_custom = Sampler(0.5, 0.7, 123)
        @test sampler_custom.temperature == Float32(0.5)
        @test sampler_custom.topp == Float32(0.7)
    end

    @testset "Softmax Tests" begin
        logits = Float32[-1.2, 2.4, 8.5]
        probs = softmax(logits)
        @test sum(probs) ≈ 1.0
    end

    @testset "Sampling Tests" begin
        @testset "sample_argmax" begin
            logits = Float32[0.1, 0.5, 0.9, 8.9, 4.5, 1.87, 0.001]
            argmax_index = sample_argmax(logits)
            @test argmax_index == 4  # Index of the highest prob
        end

        @testset "sample_mult" begin
            probabilities = Float32[0.03, 0.07, 0.2, 0.05, 0.15, 0.2, 0.3]
            coin = Float32(0.42)  # coin lies between cumulative probs of 0.35 and 0.5
            sampled_index = sample_mult(probabilities, coin)
            @test sampled_index == 5  # Should select the 5th element
        end

        @testset "sample_topp" begin
            probabilities = Float32[0.03, 0.07, 0.1, 0.05, 0.25, 0.1, 0.2, 0.08, 0.12]
            coin = Float32(0.4)
            topp = Float32(0.7) # tops probability threshold
            # 0.7 * 0.4 = 0.28 --> 0.25 < 0.28 < 0.45 --> 7th element
            sampled_index = sample_topp(probabilities, topp, coin)
            @test sampled_index == 7  # Should select the 7th element
        end
    end
    @testset "Calling sampler object" begin
        # Create sampler objects
        sampler1 = Sampler(0.0, 0.5, 187) # argmax
        sampler2 = Sampler(0.5, 1.0, 420) # multinomial
        sampler3 = Sampler(1.0, 0.7, 69) # topp

        # Define logits
        logits1 = Float32[0.278, 0.574, -0.093, 0.827, -0.641, 0.389, 0.152, 0.965, -0.738, 0.440]
        logits2 = Float32[0.761, 0.315, 0.529, -0.972, 0.148, -0.624, 0.873, -0.201, 0.456, -0.987]
        logits3 = rand(Float32, 100)

        # Call the sampler on logits

        sampled_index_11 = sampler1(logits1)
        @test sampled_index_11 == 8  
        sampled_index_12 = sampler1(logits2)
        @test sampled_index_12 == 7  
        sampled_index_13 = sampler1(logits3)
        @test 1 <= sampled_index_13 <= length(logits3)  

        sampled_index_21 = sampler2(logits1)
        @test sampled_index_21 == 1 
        sampled_index_22 = sampler2(logits2)
        @test sampled_index_22 == 9
        sampled_index_23 = sampler2(logits3)
        @test 1 <= sampled_index_23 <= length(logits3)   

        sampled_index_31 = sampler3(logits1)
        @test sampled_index_31 <= 8
        sampled_index_32 = sampler3(logits2)
        @test sampled_index_32 == 3
        sampled_index_33 = sampler3(logits3)
        @test 1 <= sampled_index_33 <= length(logits3)  

    end
end