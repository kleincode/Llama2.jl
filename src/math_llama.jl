"""
    math_llama.jl

    This file contains the math functions used in the Llama model.
"""

"""
    rmsnorm(o::Vector{Float32}, x::Vector{Float32}, weight::Vector{Float32})

Calculate the root mean square norm of a vector. 
Reference in llama2.c lines 182-195
"""
function rmsnorm!(o::AbstractArray{T}, x::AbstractArray{T}, weight::AbstractArray{T}) where T
    if length(x) == 0
        return
    end
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / length(x)
    sum_squares += 1.0f-5
    sum_squares = 1.0f0 / sqrt(sum_squares)

    o .= weight .* (sum_squares .* x)
    return nothing
end

"""
    softmax(x::AbstractArray{T})

Calculate the softmax of a vector. 
Reference in llama2.c lines 197-215
"""
function softmax!(x::AbstractArray{T}) where T
    if length(x) == 0
        return
    end

    # exp and sum
    sum_exp = exp.(x .- maximum(x))

    # normalize and update x in place
    x .= sum_exp ./ sum(sum_exp)
    
    return nothing
end

"""
    swiglu(x::Vector{Float32}, x2::Vector{Float32})
Activation function that combines GLU and Swish functions. 
```math
swiglu(x, x_2) = x * x_2 * sigmoid(x)
```
Reference in llama2.c lines 338-345
"""
function swiglu!(x::AbstractArray{T}, x2::AbstractArray{T}) where T
    if length(x) == 0
        return
    end
    sigmoid = 1 ./ (1 .+ exp.(-x))
    # SiLu function
    silu = x .* sigmoid
    x .= silu .* x2
    return nothing
end
