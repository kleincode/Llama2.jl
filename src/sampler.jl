using Random

"""
$(TYPEDEF)

Used when sorting probabilities during top-p sampling
"""
struct ProbIndex{T<:Real}
    prob::T
    index::Integer
end

"""
$(TYPEDEF)
    Sampler()
    function Sampler{T}(temperature::T, topp::T, rng_seed::Integer) where {T<:Real}

Used to return a sampled token (index) based on given logits.
Depending on the parameters, the sampler supports greedy argmax, multinomial, or top-p sampling.
It is recommended to either adjust the temperature or top-p to a non-default value but not both since they do similar things (constrain the sampling).

## Fields
$(TYPEDFIELDS)

llama2.c correspondence: Sampler (l. 577 - 715)

## Example
```julia-repl
julia> sampler_mult = Sampler{Float64}(0.5, 0.0, 1)
Sampler{Float64}(0.5, 0.0, Random.MersenneTwister(1))

julia> [sampler_mult([-0.5, 0.5, 0.2]) for i in 1:10]
10-element Vector{Int64}:
 2
 2
 2
 1
 2
 2
 3
 3
 2
 3

julia> sampler_det = Sampler{Float64}(0.0, 0.0, 1)
Sampler{Float64}(0.0, 0.0, Random.MersenneTwister(42))

julia> [sampler_det([-0.5, 0.5, 0.2]) for i in 1:10]
10-element Vector{Int64}:
 2
 2
 2
 2
 2
 2
 2
 2
 2
 2

julia> sampler_topp = Sampler{Float64}(1.0, 0.5, 1)
Sampler{Float64}(1.0, 0.5, Random.MersenneTwister(1))

julia> [sampler_topp([-0.5, 0.5, 0.2]) for i in 1:10]
10-element Vector{Int64}:
 2
 2
 2
 2
 2
 2
 3
 3
 2
 3
```
"""
struct Sampler{T<:Real}
    """Logits are divided by this value.
    A higher temperature value makes the output more diverse while a lower temperature
    makes the output more deterministic, converging to greedy argmax sampling at 0."""
    temperature::T
    """Used for top-p sampling. Only consider the set of most likely tokens whose probabilities sum up to this value.
    If this is 0 or 1, no top-p sampling is used. For other values, this prevents less likely tokens from being sampled."""
    topp::T
    rng_state::MersenneTwister

    function Sampler{T}(temperature::T, topp::T, rng_seed::Integer) where {T<:Real}
        0.0 <= temperature || throw(ArgumentError("Temperature must be non-negative"))
        0.0 <= topp <= 1.0 || throw(ArgumentError("Top-p must be in [0, 1]"))
        return new{T}(temperature, topp, MersenneTwister(rng_seed))
    end
end

Sampler() = Sampler{Float32}(1.0f0, 0.0f0, 420)

"""
$(TYPEDSIGNATURES)

Sample the next token id based on the logits.

The sampling strategy is selected based on the `temperature` and `topp` parameters of the [`Sampler`](@ref):
* If `temperature == 0`, always take the token with the highest probability (greedy argmax sampling), see [`sample_argmax`](@ref).
* If `topp` is 0 or 1, apply the temperature to the logits and sample from the predicted probability distribution (multinomial sampling), see [`sample_mult`](@ref).
* Otherwise, only sample from the smallest set of most likely tokens whose probabilities sum up to at least `topp` (top-p sampling), see [`sample_topp`](@ref). The temperature is still applied before.
"""
function (sampler::Sampler{T})(logits::Vector{T}) where {T<:Real}
    # sample the token given the logits and some hyperparameters
    next = missing
    if sampler.temperature == 0.0f0
        # greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits)
    else
        # apply the temperature to the logits
        logits = logits ./ sampler.temperature
        # apply softmax to the logits to get the probabilities for next token
        probs = softmax(logits)
        # flip a (float) coin (this is our source of entropy for sampling)
        coin = rand(sampler.rng_state, T)
        # we sample from this distribution to get the next token
        if sampler.topp == 0.0f0 || sampler.topp == 1.0f0
            # simply sample from the predicted probability distribution
            next = sample_mult(probs, coin)
        else
            # top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(probs, sampler.topp, coin)
        end
    end
    return next
end

"""
$(TYPEDSIGNATURES)

Deterministically sample the token with the highest probability.

## Example
```julia-repl
julia> sample_argmax([-0.5, 0.0, 0.5])
3
```
"""
function sample_argmax(logits::AbstractVector{T}) where {T<:Real}
    return argmax(logits)
end

"""
$(TYPEDSIGNATURES)

Sample index from a probability distribution (must sum to 1).
Coin is a random number in [0, 1).
Find the index that coin falls into.

## Examples
```julia-repl
julia> sample_mult([0.1, 0.2, 0.3, 0.4], 0.05)
1

julia> sample_mult([0.1, 0.2, 0.3, 0.4], 0.15)
2

julia> sample_mult([0.1, 0.2, 0.3, 0.4], 0.8)
4
```
"""
function sample_mult(probabilities::AbstractVector{T}, coin::T) where {T<:Real}
    sum(probabilities) ≈ 1.0 || throw(ArgumentError("Probabilities must sum to 1"))
    0.0 <= coin <= 1.0 || throw(ArgumentError("Coin must be in [0, 1]"))
    cum_prob = 0.0
    for (index, prob) in enumerate(probabilities)
        cum_prob += prob
        if coin <= cum_prob
            return index
        end
    end
    return length(probabilities) # in case of rounding errors
end

"""
$(TYPEDSIGNATURES)

Top-p sampling (or "nucleus sampling") samples from the smallest set of
tokens that exceed probability topp. This way we never sample tokens that
have very low probabilities and are less likely to go "off the rails".
Coin is a random number in [0, 1).

## Examples
```julia-repl
julia> sample_topp([0.1, 0.2, 0.3, 0.4], 1.0, 0.9)
1

julia> sample_topp([0.1, 0.2, 0.3, 0.4], 0.5, 0.9)
3

julia> sample_topp([0.1, 0.2, 0.3, 0.4], 0.4, 0.9)
3

julia> sample_topp([0.1, 0.2, 0.3, 0.4], 0.39, 0.9)
4
```
"""
function sample_topp(probabilities::AbstractVector{T}, topp::T, coin::T) where {T<:Real}
    sum(probabilities) ≈ 1.0f0 || throw(ArgumentError("Probabilities must sum to 1"))
    0.0f0 <= topp <= 1.0f0 || throw(ArgumentError("Top-p must be in [0, 1]"))
    0.0f0 <= coin <= 1.0f0 || throw(ArgumentError("Coin must be in [0, 1]"))
    # values smaller than (1 - topp) / (n - 1) cannot be part of the result
    # so for efficiency we crop these out as candidates before sorting
    cutoff = (1.0f0 - topp) / (length(probabilities) - 1.0f0)
    idx_greater = findall(x -> x >= cutoff, probabilities)
    # add all probabilities that are greater than cutoff
    probindex = [ProbIndex{T}(probabilities[i], i) for i in idx_greater]
    # sort indices in descending order of probabilities
    sort!(probindex; by=x -> x.prob, rev=true)
    # truncate the list where cumulative probability exceeds topp
    cum_prob = 0.0f0
    for (i, probind) in enumerate(probindex)
        cum_prob += probind.prob
        if cum_prob > topp
            # remove all elements after this one
            probindex = @view probindex[begin:i]
            break # we've exceeded topp 
        end
    end
    # sample from the truncated list
    r = coin * cum_prob
    cdf = 0.0f0
    for probind in probindex
        cdf += probind.prob
        if r < cdf
            return probind.index
        end
    end
    return probindex[end].index # in case of rounding errors
end
