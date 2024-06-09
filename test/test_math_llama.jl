using Llama2
using Test

@testset "math_llama.jl" begin
    @testset "rmsnorm" begin
        # Test 1: Typical case
        x = Float32[1.0, 2.0, 3.0]
        weight = Float32[1.0, 1.0, 1.0]
        res_1 = rmsnorm(x, weight)
        exp_1 = [0.46290955, 0.92581911, 1.38872866]
        @test isapprox(res_1, exp_1, atol=1e-6)

        # Test 2: Case with zero values
        x = Float32[0, 0, 0, 0, 0]
        weight = Float32[1, 1, 1, 1, 1]
        res_2 = rmsnorm(x, weight)
        exp_2 = [0, 0, 0, 0, 0]
        @test isapprox(res_2, exp_2, atol=1e-6)

        # Test 3: Case with negative values
        x = Float32[-1, -2, -3, -4, -5]
        weight = Float32[1, 1, 1, 1, 1]
        res_3 = rmsnorm(x, weight)
        exp_3 = [-0.30151121, -0.60302242, -0.90453362, -1.20604483, -1.50755604]
        @test isapprox(res_3, exp_3, atol=1e-6)

        # Test 4: Case with very small values
        x = Float32[1e-6, 2e-6, 3e-6]
        weight = Float32[1, 1, 2]
        res_4 = rmsnorm(x, weight)
        exp_4 = [0.00031622764, 0.00063245529, 0.00189736588]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 5: Edge case with single element
        x = Float32[3.0]
        weight = Float32[2.0]
        res_5 = rmsnorm(x, weight)
        exp_5 = [1.999998888889815]
        @test res_5 ≈ exp_5 

        # Test 6: Case with empty vector
        x = Float32[]
        weight = Float32[]
        res_6 = rmsnorm(x, weight)
        exp_6 = Float32[]
        @test res_6 == exp_6

    end


    @testset "softmax" begin
        # Test 1: Typical case
        x = Float32[1.0, 2.0, 3.0]
        res_1 = softmax(x)
        exp_1 = [0.09003057, 0.24472847, 0.66524096]
        @test isapprox(res_1, exp_1, atol=1e-6)

        # Test 2: Case zero vector
        x = Float32[0.0, 0.0, 0.0]
        res_2 = softmax(x)
        exp_2 = [0.33333333, 0.33333333, 0.33333333]
        @test isapprox(res_2, exp_2, atol=1e-6)

        # Test 3: Case with very small values
        x = Float32[1e-6, 2e-6, 3e-6]
        res_3 = softmax(x) 
        exp_3 = [0.333333, 0.33333333, 0.33333367]
        @test isapprox(res_3, exp_3, atol=1e-6)

        # Test 4: Case with very large values
        x = Float32[1e6, 2e6, 3e6]
        res_4 = softmax(x)
        exp_4 = [0.0, 0.0, 1.0]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 5: Edge case with single element
        x = Float32[23.0]
        res_5 = softmax(x)
        exp_5 = [1.0]
        @test isapprox(res_4, exp_4, atol=1e-6)

        # Test 6: Case with negative values
        x = Float32[-1.0, -2.0, -3.0]
        res_6 = softmax(x)
        exp_6 = [0.66524096, 0.24472847, 0.09003057]
        @test isapprox(res_6, exp_6, atol=1e-6)

        # Test 7: Case with empty vector
        x = Float32[]
        res_7 = softmax(x)
        exp_7 = Float32[]
        @test res_7 == exp_7
    end

    @testset "swiglu" begin
        # Test 1: Typical case
        x = Float32[1.0, 2.0, 3.0]
        x2 = Float32[1.0, 2.0, 3.0]
        res_1 = swiglu(x, x2)
        exp_1 = [0.73105858, 3.52318831, 8.57316714]
        @test isapprox(res_1, exp_1, atol=1e-6)

        # Test 2: Case with zero values
        x = Float32[0.0, 0.0, 0.0]
        x2 = Float32[1.0, 2.0, 3.0]
        res_2 = swiglu(x, x2)
        exp_2 = [0.0, 0.0, 0.0]
        @test isapprox(res_2, exp_2, atol=1e-6)

        # Test 3: Case with negative values
        x = Float32[-1.0, 0.0, 1.0]
        x2 = Float32[3.0, -2.0, 1.0]
        res_3 = swiglu(x, x2)
        exp_3 = [-0.80682426, 0, 0.73105858]
        @test isapprox(res_3, exp_3, atol=1e-6)

        # Test 4: Case with very small values
        x = Float32[1e-3, 2e-3, 3e-3]
        x2 = Float32[1e-3, 2e-3, 3e-3]
        res_4 = swiglu(x, x2)
        exp_4 = [5.00250000e-07, 2.00200000e-06, 4.50674999e-06]
        @test isapprox(res_4, exp_4)

        # Test 5: Edge case with single element
        x = Float32[3.0]
        x2 = Float32[2.0]
        res_5 = swiglu(x, x2)
        exp_5 = [5.71544476]
        @test isapprox(res_5, exp_5, atol=1e-6)

        # Test 6: Case with empty vector
        x = Float32[]
        x2 = Float32[]
        res_6 = swiglu(x, x2)
        exp_6 = Float32[]
        @test res_6 == exp_6
    end
end

