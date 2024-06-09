using Llama2
using Test

@testset "Llama2.jl" begin
    # Write your tests here.
    @test timesthree(2) == 6
    include("test_math_llama.jl")
end
