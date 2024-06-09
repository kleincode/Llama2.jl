"""
    math_llama.jl

    This file contains the math functions used in the Llama model.
"""

"""
    rmsnorm(x::Vector{Float32}, weight::Vector{Float32})

Calculate the root mean square norm of a vector. 
Reference in llama2.c lines 182-195
"""
function rmsnorm(x::Vector{Float32}, weight::Vector{Float32})
    if length(x) == 0
        return Float32[]
    end
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / length(x)
    sum_squares += 1.0f-5
    sum_squares = 1.0f0 / sqrt(sum_squares) 
    
    return weight.*(sum_squares.*x)
end


"""
    softmax(x::Vector{Float32})

Calculate the softmax of a vector. 
Reference in llama2.c lines 197-215
"""
function softmax(x::Vector{Float32})::Vector{Float32}
    if length(x) == 0
        return Float32[]
    end

    # exp and sum
    sum_exp = exp.(x.-maximum(x))

    # normalize and return

    return sum_exp./sum(sum_exp)
end


"""
    swiglu(x::Vector{Float32}, x2::Vector{Float32})
Activation function that combines GLU and Swish functions. 
Reference in llama2.c lines 338-345
"""
function swiglu(x::Vector{Float32}, x2::Vector{Float32})::Vector{Float32}
    if length(x) == 0
        return Float32[]
    end
    log_sigmoid = 1 ./ (1 .+ exp.(-x))
    # SiLu function
    silu = x .* log_sigmoid
    
    return silu .* x2
end
