"""
    math_llama.jl

    This file contains the math functions used in the Llama model.
"""

"""
$(TYPEDSIGNATURES)

Calculate the root mean square norm of a vector. 
Reference in llama2.c lines 182-195
"""
function rmsnorm!(
    o::AbstractArray{T}, x::AbstractArray{T}, weight::AbstractArray{T}
) where {T<:Real}
    if length(x) == 0
        return nothing
    end
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / length(x) + 1.0f-5
    sum_squares = 1.0f0 / sqrt(sum_squares)

    o .= weight .* (sum_squares .* x)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Calculate the softmax of a vector. 
Reference in llama2.c lines 197-215
"""
function softmax!(x::AbstractArray{T}) where {T<:Real}
    if length(x) == 0
        return nothing
    end

    # exp and sum
    x .= exp.(x .- maximum(x))

    # normalize and update x in place
    x ./= sum(x)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Activation function that combines GLU and Swish functions. 
```math
swiglu(x, x_2) = x * x_2 * sigmoid(x)
```
Reference in llama2.c lines 338-345
"""
function swiglu!(x::AbstractArray{T}, x2::AbstractArray{T}) where {T<:Real}
    if length(x) == 0
        return nothing
    end
    x .*= x2 ./ (1 .+ exp.(-x))
    return nothing
end
