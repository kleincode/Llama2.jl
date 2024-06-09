begin
# Calculate the root mean square norm of a vector
function rmsnorm(x::Vector, weight::Vector)
    size = length(x)
    # Calculate the sum of the squares
    sum_squares = sum(x .^ 2) / size
    sum_squares += 1e-5
    sum_squares = 1.0 / sqrt(sum_squares) 
    
    o = weight .* (sum_squares .* x)
    
    return o
end


# Calculate the softmax of a vector
function softmax(x::Vector)::Vector{Float32}

    # exp and sum
    sum_exp = exp.(x .- maximum(x))

    # normalize
    res = sum_exp ./ sum(sum_exp)

    return res
end

end