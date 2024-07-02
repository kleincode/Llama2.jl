"""
    math_llama.jl

    This file contains the math functions used in the Llama model.
"""

"""
$(TYPEDSIGNATURES)

Calculate the root mean square norm of a vector. 
Reference in llama2.c lines 182-195
"""
function rmsnorm(x::AbstractVector{T}, weight::AbstractVector{T}) where {T<:Real}
    if length(x) == 0
        return T[]
    end
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / length(x)
    sum_squares += 1.0f-5
    sum_squares = 1.0f0 / sqrt(sum_squares)

    return weight .* (sum_squares .* x)
end

"""
$(TYPEDSIGNATURES)

Calculate the softmax of a vector. 
Reference in llama2.c lines 197-215
"""
function softmax(x::AbstractVector{T}) where {T<:Real}
    if length(x) == 0
        return T[]
    end

    # exp and sum
    sum_exp = exp.(x .- maximum(x))

    # normalize and return

    return sum_exp ./ sum(sum_exp)
end

"""
$(TYPEDSIGNATURES)

Activation function that combines GLU and Swish functions. 
```math
swiglu(x, x_2) = x * x_2 * sigmoid(x)
```
Reference in llama2.c lines 338-345
"""
function swiglu(x::AbstractVector{T}, x2::AbstractVector{T}) where {T<:Real}
    if length(x) == 0
        return T[]
    end
    sigmoid = 1 ./ (1 .+ exp.(-x))
    # SiLu function
    silu = x .* sigmoid
    return silu .* x2
end
