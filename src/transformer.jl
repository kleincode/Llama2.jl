using LinearAlgebra

"""
$(TYPEDEF)
    function TransformerWeights(config::Config) where {T<:Real}

Holds the weights for the Llama2 transformer model.

## Fields
$(TYPEDFIELDS)

llama2.c correspondence: TransformerWeights (l. 29)

## Allocate from config
To create a new `TransformerWeights` instance with preallocated matrices, use the config constructor:

    function TransformerWeights(config::Config) where {T<:Real}

llama2.c correspondence: memory_map_weights (l. 111)
"""
struct TransformerWeights{T<:Real}
    """Token embedding table: Mapping from token index to embedding vector. Shape: (dim, vocab_size)"""
    token_embedding_table::Matrix{T}

    # weights
    """Weights for rmsnorm before the attention for each layer. Shape: (dim, n_layers)"""
    rms_att_weight::Matrix{T}
    """Weights for rmsnorm before the feed-forward net for each layer. Shape: (dim, n_layers)"""
    rms_ffn_weight::Matrix{T}

    # weights for matmuls (dim == n_heads * head_size)
    """Query weights for each attention layer. Shape: (n_heads * head_size, dim, n_layers)"""
    wq::Array{T,3}
    """Key weights for each attention layer. Shape: (dim, kv_dim, n_layers)"""
    wk::Array{T,3}  # llama2.c says (kv_dim, dim, n_layers), should be (dim, kv_dim, n_layers)
    """Value weights for each attention layer. Shape: (dim, kv_dim, n_layers)"""
    wv::Array{T,3}  # llama2.c says (kv_dim, dim, n_layers), should be (dim, kv_dim, n_layers)
    """Output weights for each attention layer. Shape: (n_heads * head_size, dim, n_layers)"""
    wo::Array{T,3}  # llama2.c says (dim, n_heads * head_size, n_layers), should be (n_heads * head_size, dim, n_layers)

    # weights for ffn
    """First weight matrix for each feed forward layer (in -> hidden). Shape: (dim, hidden_dim, n_layers)"""
    w1::Array{T,3}
    """Second weight matrix for each feed forward layer (hidden -> out). Shape: (hidden_dim, dim, n_layers)"""
    w2::Array{T,3}
    """Third weight matrix for each feed forward layer (in -> hidden). Shape: (dim, hidden_dim, n_layers)"""
    w3::Array{T,3}

    # final rmsnorm
    """Weights for the final rmsnorm before the optional classifier head. Shape: (dim,)"""
    rms_final_weight::Vector{T}

    # optional classifier weights
    """Weights for the optional classifier head. If there is no classifier (the usual case), this should equal token_embedding_table, translating embeddings back to logits. This is inspired by the original llama2.c implementation. Shape: (dim, vocab_size)"""
    wcls::Matrix{Float32}  # (dim, vocab_size)
end

function TransformerWeights{T}(config::Config) where {T<:Real}
    (; dim, hidden_dim, n_heads, n_kv_heads, n_layers, vocab_size) = config
    kv_dim = (dim * n_kv_heads) ÷ n_heads

    # initialization of undefined arrays
    return TransformerWeights{T}(
        Matrix{T}(undef, dim, vocab_size),
        Matrix{T}(undef, dim, n_layers),
        Matrix{T}(undef, dim, n_layers),
        Array{T}(undef, dim, dim, n_layers),
        Array{T}(undef, dim, kv_dim, n_layers),
        Array{T}(undef, dim, kv_dim, n_layers),
        Array{T}(undef, dim, dim, n_layers),
        Array{T}(undef, dim, hidden_dim, n_layers),
        Array{T}(undef, hidden_dim, dim, n_layers),
        Array{T}(undef, dim, hidden_dim, n_layers),
        Vector{T}(undef, config.dim),
        Matrix{T}(undef, dim, vocab_size),
    )
end

"""
$(TYPEDEF)

State of the transformer model. The matrices are modified during a forward pass.
It should never be necessary to manually modify this.
While some of these arrays preserve actual neccessary state, some of them serve as preallocated buffers to speed up computation in the [`forward!`](@ref) method.

## Fields
$(TYPEDFIELDS)

llama2.c correspondence: RunState (l. 50)

## Allocate from config
    function RunState(config::Config) where {T<:Real}
Initializes the matrices in RunState based on the shapes provided in the Config.
"""
mutable struct RunState{T<:Real}
    # current way of activations
    """Activations at current time stamp. Shape: (dim,)"""
    x::Vector{T}
    """Activations at current time stamp inside a residual branch. Shape: (dim,)"""
    xb::Vector{T}
    """An additional activation buffer for convenience. Shape: (dim,)"""
    xb2::Vector{T}
    """Buffer for the hidden dimension in the feed-forward net. Shape: (hidden_dim,)"""
    hb::Vector{T}
    """Buffer for the hidden dimension in the feed-forward net. Shape: (hidden_dim,)"""
    hb2::Vector{T}
    """Stores the query vector in the attention part. Shape: (n_heads * head_size,)"""
    q::Vector{T}
    # k::Vector{T}   # key (dim,) - this is just a pointer to a portion of key_cache
    # v::Vector{T}   # value (dim,) - this is just a pointer to a portion of value_cache
    """Buffer for the attention scores. Shape: (n_heads, seq_len)"""
    att::Matrix{T}
    """The output logits. Shape: (vocab_size,)"""
    logits::Vector{T}

    # kv cache (n_kv_heads * head_size == kv_dim)
    """Cache for all the keys in the attention part. Shape: (n_kv_heads * head_size, seq_len, n_layers)"""
    key_cache::Array{T,3}
    """Cache for all the values in the attention part. Shape: (n_kv_heads * head_size, seq_len, n_layers)"""
    value_cache::Array{T,3}
end

function RunState{T}(config::Config) where {T<:Real}
    # calculation of kv_dim
    kv_dim::Int32 = (config.dim * config.n_kv_heads) ÷ config.n_heads

    # initialization of undefined arrays
    return RunState{T}(
        Array{T}(undef, config.dim), # x
        Array{T}(undef, config.dim), # xb
        Array{T}(undef, config.dim), # xb2
        Array{T}(undef, config.hidden_dim), # hb
        Array{T}(undef, config.hidden_dim), # hb2
        Array{T}(undef, config.dim),  # q
        Array{T}(undef, config.n_heads, config.seq_len),  # att
        Array{T}(undef, config.vocab_size),   # logits
        Array{T}(undef, kv_dim, config.seq_len, config.n_layers), # key_cache
        Array{T}(undef, kv_dim, config.seq_len, config.n_layers), # value_cache
    )
end

"""
$(TYPEDEF)

A transformer model, consisting of a config, weights, and a run state.

## Fields
$(TYPEDFIELDS)

llama2.c correspondence: Transformer (l. 67)
"""
struct Transformer{T<:Real}
    """Hyperparameters of the architecture"""
    config::Config
    """Weights of the module"""
    weights::TransformerWeights{T}
    """Buffers for the wave of activations in the forward pass"""
    state::RunState{T}
end

"""
$(TYPEDSIGNATURES)

A single complete transformer forward pass for input token `token` at position `pos`, returning the output logits.
`pos` is one-based, i.e. 1 <= pos <= seq_len.
`token` is also a one-based token index.
This modifies the RunState of the transformer.

llama2.c correspondence: forward (l. 231)

## Example
To run token 5 at position 1 through the transformer and get the predicted output logits:
```julia-repl
julia> forward!(transformer, 5, 1)
32000-element Vector{Float32}:
 -2.1009917
  1.664739
 -2.1005554
 -2.1007848
 -2.1005578
 -2.1009412
  ⋮
 -2.1007295
 -2.100759
 -2.1007874
 -2.1009996
 -2.1009269
 -2.1007652
```
"""
@views function forward!(
    transformer::Transformer{T}, token::Integer, pos::Integer
) where {T<:Real}
    # a few convenience variables
    (; dim, n_heads, n_kv_heads, n_layers, seq_len) = transformer.config
    1 <= pos <= seq_len || throw(ArgumentError("1 <= pos <= seq_len is required"))
    s = transformer.state
    w = transformer.weights
    kv_dim = (dim * n_kv_heads) ÷ n_heads
    kv_mul = n_heads ÷ n_kv_heads # integer multiplier of the kv sharing in multiquery
    head_size = dim ÷ n_heads

    # copy the token embedding into x
    s.x .= w.token_embedding_table[:, token] # (dim,)

    # forward all the layers
    for l in 1:n_layers
        # attention rmsnorm
        rmsnorm!(s.xb, s.x, w.rms_att_weight[:, l]) # (n_heads * head_size,)

        # qkv matmuls for this position
        mul!(s.q, w.wq[:, :, l]', s.xb) # (n_heads * head_size,) = (dim, n_heads * head_size) * (n_heads * head_size,)
        mul!(s.key_cache[:, pos, l], w.wk[:, :, l]', s.xb) # (kv_dim,) = (kv_dim, dim) * (n_heads * head_size,)
        mul!(s.value_cache[:, pos, l], w.wv[:, :, l]', s.xb) # (kv_dim,) = (kv_dim, dim) * (n_heads * head_size,)

        # RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in 1:2:dim
            head_dim = (i - 1) % head_size
            freq = 1.0f0 / (10000.0f0^(head_dim / head_size))
            val = (pos - 1) * freq
            fcr = cos(val)
            fci = sin(val)

            # rotate query
            v0 = s.q[i]
            v1 = s.q[i + 1]
            s.q[i] = v0 * fcr - v1 * fci
            s.q[i + 1] = v0 * fci + v1 * fcr

            # rotate key only if i <= kv_dim
            if i <= kv_dim
                v0 = s.key_cache[i, pos, l]
                v1 = s.key_cache[i + 1, pos, l]
                s.key_cache[i, pos, l] = v0 * fcr - v1 * fci
                s.key_cache[i + 1, pos, l] = v0 * fci + v1 * fcr
            end
        end

        # iterate over all heads
        for h in 1:n_heads
            # get the query vector for this head
            h_off = (h - 1) * head_size
            q = s.q[(h_off + 1):(h_off + head_size)] # (head_size,)
            # iterate over all timesteps, including the current one
            kv_ind = ((h - 1) ÷ kv_mul) * head_size
            for t in 1:pos
                k = s.key_cache[(kv_ind + 1):(kv_ind + head_size), t, l] # (head_size,)
                score = (q ⋅ k) / sqrt(Float32(head_size)) # scalar
                s.att[h, t] = score
            end
            # softmax the scores to get attention weights, from 1..pos inclusively
            softmax!(s.att[h, 1:pos]) # (pos,)
            # weighted sum of the values, store back into xb
            s.xb[(h_off + 1):(h_off + head_size)] .= 0 # (head_size,)
            for t in 1:pos
                # get the value vector for this head and at this timestep
                v = s.value_cache[(kv_ind + 1):(kv_ind + head_size), t, l] # (head_size,)
                # accumulate the weighted value into xb
                s.xb[(h_off + 1):(h_off + head_size)] += s.att[h, t] * v # (head_size,) = scalar * (head_size,)
            end
        end

        # final matmul to get the output of the attention
        mul!(s.xb2, w.wo[:, :, l]', s.xb) # (dim,) = (dim, n_heads * head_size) * (n_heads * head_size,)

        # residual connection back into x
        s.x .+= s.xb2 # (dim,) += (dim,)

        # ffn rmsnorm
        rmsnorm!(s.xb, s.x, w.rms_ffn_weight[:, l]) # (dim,)

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        mul!(s.hb, w.w1[:, :, l]', s.xb) # (hidden_dim,) = (hidden_dim, dim) * (dim,)
        mul!(s.hb2, w.w3[:, :, l]', s.xb) # (hidden_dim,) = (hidden_dim, dim) * (dim,)

        # SwiGLU non-linearity
        swiglu!(s.hb, s.hb2) # (hidden_dim,)
        # final matmul to get the output of the ffn
        mul!(s.xb, w.w2[:, :, l]', s.hb) # (dim,) = (dim, hidden_dim) * (hidden_dim,)
        # residual connection
        s.x .+= s.xb # (dim,)
    end

    # final rmsnorm
    rmsnorm!(s.x, s.x, w.rms_final_weight)
    # classifier into logits
    mul!(s.logits, w.wcls', s.x) # (vocab_size,) = (vocab_size, dim) * (dim,)
    return s.logits
end