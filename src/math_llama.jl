"""
    math_llama.jl

    This file contains the math functions used in the Llama model.
"""

begin
# Calculate the root mean square norm of a vector. Reference in llama2.c lines 182-195
function rmsnorm(x::Vector{Float32}, weight::Vector{Float32})
    size = length(x)
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / size
    sum_squares += 1e-5
    sum_squares = 1.0 / sqrt(sum_squares) 
    
    o = weight .* (sum_squares .* x)
    
    return o
end


# Calculate the softmax of a vector. Reference in llama2.c lines 197-215
function softmax(x::Vector{Float32})::Vector{Float32}

    # exp and sum
    sum_exp = exp.(x .- maximum(x))

    # normalize
    res = sum_exp ./ sum(sum_exp)

    return res
end


# Activation function that combines GLU and Swish functions. Reference in llama2.c lines 338-345
function swiglu(x::Vector{Float32}, x2::Vector{Float32})::Vector{Float32}
    log_sigmoid = 1 ./ (1 .+ exp.(-x))
    # SiLu function
    silu = x .* log_sigmoid
    
    return silu .* x2
end

end