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
        logits = Float32[-1.2, 1.2, 4.8, 2.4]
        probs = softmax(logits)
        @test sum(probs) ≈ 1.0
        @test probs ≈ Float32[0.00221214, 0.02438485, 0.89244245, 0.08096055]
    end

    @testset "Sampling Tests" begin
        @testset "sample_argmax" begin
            logits = Float32[0.1, 0.5, 0.9, 8.9, 4.5, 1.87, 0.001]
            argmax_index = sample_argmax(logits)
            @test argmax_index == 4  # Index of the highest prob
        end

        probabilities1 = Float32[0.03, 0.0, 0.07, 0.2, 0.05, 0.15, 0.2, 0.3]
        probabilities2 = Float32[0.03, 0.07, 0.2, 0.05, 0.15, 0.2, 0.3, 0.0]
        probabilities3 = Float32[1.0]
        probabilities_gt1 = Float32[0.5, 0.3, 0.21]
        probabilities_lt1 = Float32[0.2, 0.3, 0.49998]

        coin1 = Float32(0.4)
        coin2 = Float32(0.1)
        coin3 = Float32(0.0)
        coin4 = Float32(0.99999)
        @testset "sample_mult" begin
            sampled_index_11 = sample_mult(probabilities1, coin1)
            @test sampled_index_11 == 6
            sampled_index_22 = sample_mult(probabilities2, coin2)
            @test sampled_index_22 == 2
            sampled_index_23 = sample_mult(probabilities2, coin3)
            @test sampled_index_23 == 1
            sampled_index_14 = sample_mult(probabilities1, coin4)
            @test sampled_index_14 == 8
            sampled_index_24 = sample_mult(probabilities2, coin4)
            @test sampled_index_24 == 7
            sampled_index_34 = sample_mult(probabilities3, coin4)
            @test sampled_index_34 == 1
            # Sum > 1 -> throw
            @test_throws ArgumentError sample_mult(probabilities_gt1, coin4)
            # Sum ≈ 1 but Sum < Coin --> return last element
            sampled_index_lt1 = sample_mult(probabilities_lt1, coin4)
            @test sampled_index_lt1 == length(probabilities_lt1)
        end

        topp1 = Float32(0.8)
        topp2 = Float32(0.0)
        topp3 = Float32(1.0)
        topp4 = Float32(-11.11)

        @testset "sample_topp" begin
            sampled_index_111 = sample_topp(probabilities1, topp1, coin1)
            # 0.8 * 0.4 = 0.32 --> 0.3 < 0.32 < 0.5 --> 4th element
            @test sampled_index_111 == 4
            sampled_index_214 = sample_topp(probabilities2, topp1, coin4)
            # 0.8 * 0.99999 ≈ 0.8 --> 0.7 < 0.8 < 0.85 --> 5th element
            @test sampled_index_214 == 5
            sampled_index_122 = sample_topp(probabilities1, topp2, coin2)
            @test sampled_index_122 == 8
            sampled_index_234 = sample_topp(probabilities2, topp3, coin4)
            @test sampled_index_234 == 1
            @test_throws ArgumentError sample_topp(probabilities3, topp4, coin4)
            # Sum ≈ 1 but Sum < Coin --> return last element
            sampled_index_lt1 = sample_topp(probabilities_lt1, topp1, coin4)
            @test sampled_index_lt1 == 1 # last element after sort
        end
    end
    @testset "Calling sampler object" begin
        # Create sampler objects
        sampler1 = Sampler(0.0, 0.5, 187) # argmax
        sampler2 = Sampler(0.5, 1.0, 420) # multinomial
        sampler3 = Sampler(1.0, 0.7, 69) # topp

        # Define logits
        logits1 = Float32[
            0.278, 0.574, -0.093, 0.827, -0.641, 0.389, 0.152, 0.965, -0.738, 0.440
        ]
        logits2 = Float32[
            0.761, 0.315, 0.529, -0.972, 0.148, -0.624, 0.873, -0.201, 0.456, -0.987
        ]
        logits3 = rand(Float32, 100)

        # Call the sampler on logits

        sampled_index_11 = sampler1(logits1)
        @test sampled_index_11 == 8
        sampled_index_12 = sampler1(logits2)
        @test sampled_index_12 == 7
        sampled_index_13 = sampler1(logits3)
        @test 1 <= sampled_index_13 <= length(logits3)

        sampled_index_21 = sampler2(logits1)
        @test sampled_index_21 == 10
        sampled_index_22 = sampler2(logits2)
        @test sampled_index_22 == 1
        sampled_index_23 = sampler2(logits3)
        @test 1 <= sampled_index_23 <= length(logits3)

        sampled_index_31 = sampler3(logits1)
        @test sampled_index_31 == 10
        sampled_index_32 = sampler3(logits2)
        @test sampled_index_32 == 2
        sampled_index_33 = sampler3(logits3)
        @test 1 <= sampled_index_33 <= length(logits3)
    end
end