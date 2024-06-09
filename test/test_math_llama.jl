using Llama2
using Test

@testset "math_llama.jl" begin
    @testset "rmsnorm" begin
        # Test 1: Typical case
        x = [1.0, 2.0, 3.0]
        weight = [1.0, 1.0, 1.0]
        res_1 = rmsnorm(x, weight)
        exp_1 = [0.46290955, 0.92581911, 1.38872866]
        @test isapprox(res_1, exp_1, atol=1e-6)

        # Test 2: Case with zero values
        x = [0, 0, 0, 0, 0]
        weight = [1, 1, 1, 1, 1]
        res_2 = rmsnorm(x, weight)
        exp_2 = [0, 0, 0, 0, 0]
        @test isapprox(res_2, exp_2, atol=1e-6)

        # Test 3: Case with negative values
        x = [-1, -2, -3, -4, -5]
        weight = [1, 1, 1, 1, 1]
        res_3 = rmsnorm(x, weight)
        exp_3 = [-0.30151121, -0.60302242, -0.90453362, -1.20604483, -1.50755604]
        @test isapprox(res_3, exp_3, atol=1e-6)

        # Test 4: Case with very small values
        x = [1e-6, 2e-6, 3e-6]
        weight = [1, 1, 2]
        res_4 = rmsnorm(x, weight)
        exp_4 = [0.00031622764, 0.00063245529, 0.00189736588]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 5: Edge case with size = 1
        x = [3.0]
        weight = [2.0]
        res_5 = rmsnorm(x, weight)
        exp_5 = [1.999998888889815]
        @test res_5 ≈ exp_5 # res_5 ≈ exp_5 is equivalent to res_5 ≈ exp_5
    end


    @testset "softmax" begin
        # Test 1: Typical case
        x = [1.0, 2.0, 3.0]
        res_1 = softmax(x)
        exp_1 = [0.09003057, 0.24472847, 0.66524096]
        @test isapprox(res_1, exp_1, atol=1e-6)

        # Test 2: Case zero vector
        x = [0.0, 0.0, 0.0]
        res_2 = softmax(x)
        exp_2 = [0.33333333, 0.33333333, 0.33333333]
        @test isapprox(res_2, exp_2, atol=1e-6)

        # Test 3: Case with very small values
        x = [1e-6, 2e-6, 3e-6]
        res_3 = softmax(x) 
        exp_3 = [0.333333, 0.33333333, 0.33333367]
        @test isapprox(res_3, exp_3, atol=1e-6)

        # Test 4: Case with very large values
        x = [1e6, 2e6, 3e6]
        res_4 = softmax(x)
        exp_4 = [0.0, 0.0, 1.0]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 5: Edge case with size = 1
        x = [23.0]
        res_5 = softmax(x)
        exp_5 = [1.0]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 6: Case with negative values
        x = [-1.0, -2.0, -3.0]
        res_6 = softmax(x)
        exp_6 = [0.66524096, 0.24472847, 0.09003057]
        @test isapprox(res_6, exp_6, atol=1e-6)
    end
end

