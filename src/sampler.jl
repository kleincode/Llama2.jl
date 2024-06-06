"""
Used to return a sampled token (index) based on given logits

llama.c correspondence: Sampler (l. 577 - 715)
"""

using Random

struct ProbIndex
    """
    Used when sorting probabilities during top-p sampling
    """
    prob::Float32
    index::Int
end

struct Sampler
    temperature::Float32
    topp::Float32
    rng_state::MersenneTwister

    function Sampler(temperature::Float64, topp::Float64, rng_seed::Int64)
        0.0 <= temperature || throw(ArgumentError("Temperature must be non-negative"))
        0.0 <= topp <= 1.0 || throw(ArgumentError("Top-p must be in [0, 1]"))
        return new(Float32(temperature), Float32(topp), MersenneTwister(rng_seed))
    end
end

function Sampler()
    """
    Create Sampler with default parameters
    """
    return Sampler(1.0, 0.0, 420)
end

function (sampler::Sampler)(logits::Vector{Float32})
    # sample the token given the logits and some hyperparameters
    next = missing
    if sampler.temperature == 0.0
        # greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits)
    else
        # apply the temperature to the logits
        logits = logits ./ sampler.temperature
        # apply softmax to the logits to get the probabilities for next token
        probs = softmax(logits)
        # flip a (float) coin (this is our source of entropy for sampling)
        coin = randn(sampler.rng_state, Float32)
        # we sample from this distribution to get the next token
        if sampler.topp == 0.0 || sampler.topp == 1.0
            # simply sample from the predicted probability distribution
            next = sample_mult(probs, coin)
        else 
            # top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(probs, sampler.topp, coin)
        end
    end 
    return next
end

function sample_argmax(logits::Vector{Float32})
    """
    Return the index that has the highest probability
    """
    return argmax(logits)
end

function sample_mult(probabilities::Vector{Float32}, coin::Float32)
    """
    Sample index from probabilities (they must sum to 1!).
    Coin is a random number in [0, 1).
    Find the index that coin falls into.
    """
    cum_prob = 0.0
    for (index, prob) in enumerate(probabilities)
        cum_prob += prob
        if coin <= cum_prob
            return index
        end
    end
    return length(probabilities) # in case of rounding errors
end

function sample_topp(probabilities::Vector{Float32}, topp::Float32, coin::Float32)
    """
    Top-p sampling (or "nucleus sampling") samples from the smallest set of
    tokens that exceed probability topp. This way we never sample tokens that
    have very low probabilities and are less likely to go "off the rails".
    Coin is a random number in [0, 1)
    """
    # values smaller than (1 - topp) / (n - 1) cannot be part of the result
    # so for efficiency we crop these out as candidates before sorting
    cutoff = (1.0 - topp) / (length(probabilities) - 1);
    idx_greater = findall(x -> x >= cutoff, probabilities)
    # add all probabilities that are greater than cutoff
    probindex = [ProbIndex(probabilities[i], i) for i in idx_greater]
    # sort indices in descending order of probabilities
    sort!(probindex, by = x -> x.prob, rev = true)
    # truncate the list where cumulative probability exceeds topp
    cum_prob = 0.0
    for probind in probindex
        cum_prob += probind.prob
        if cum_prob > topp
            # remove all elements after this one
            probindex = probindex[1:findfirst(x -> x == probind, probindex)]
            break # we've exceeded topp 
        end
    end
    # sample from the truncated list
    r = coin * cum_prob
    cdf = 0.0
    for probind in probindex
        cdf += probind.prob
        if r < cdf
            return probind.index
        end
    end
    return probindex[end].index # in case of rounding errors
end

function softmax(values::Vector{Float32})
    exp_values = exp.(values)
    return exp_values ./ sum(exp_values)
end