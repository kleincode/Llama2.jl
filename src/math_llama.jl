# Calculate the root mean square norm of a vector
function rmsnorm(o::Array, x::Array, weight::Array, size::Int)
    # Calculate the sum of the squares
    ss = 0.0
    for j in 1:size
        ss += x[j]^2
    end
    ss /= size
    ss += 1e-5
    ss = 1.0 / sqrt(ss)

    # Calculate norm and scale
    for j in 1:size
        o[j] = weight[j] * (ss * x[j])
    end
end

timestwo(x) = 2 * x
