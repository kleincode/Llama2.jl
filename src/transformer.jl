using LinearAlgebra

"""
Weights for the Llama2 transformer model.

llama2.c correspondence: TransformerWeights (l. 29)
"""
struct TransformerWeights
    # token embedding table
    token_embedding_table::Matrix{Float32} # (dim, vocab_size)

    # weights
    rms_att_weight::Matrix{Float32}         # (dim, n_layers)
    rms_ffn_weight::Matrix{Float32}         # (dim, n_layers)

    # weights for matmuls (dim == n_heads * head_size)
    wq::Array{Float32,3}  # (n_heads * head_size, dim, n_layers)
    wk::Array{Float32,3}  # llama2.c says (kv_dim, dim, n_layers), should be (dim, kv_dim, n_layers)
    wv::Array{Float32,3}  # llama2.c says (kv_dim, dim, n_layers), should be (dim, kv_dim, n_layers)
    wo::Array{Float32,3}  # llama2.c says (dim, n_heads * head_size, n_layers), should be (n_heads * head_size, dim, n_layers)

    # weights for ffn
    w1::Array{Float32,3}  # (dim, hidden_dim, n_layers)
    w2::Array{Float32,3}  # (hidden_dim, dim, n_layers)
    w3::Array{Float32,3}  # (dim, hidden_dim, n_layers)

    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)

    # optional classifier weights
    wcls::Matrix{Float32}  # (dim, vocab_size)
end

"""
State of the transformer model. Modified during a forward pass.

llama2.c correspondence: RunState (l. 50)
"""
mutable struct RunState
    # current way of activations
    x::Vector{Float32}  # activation at current time stamp (dim,)
    xb::Vector{Float32}  # same, but inside a residual branch (dim,)
    xb2::Vector{Float32}  # an additional buffer just for convenience (dim,)
    hb::Vector{Float32}  # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vector{Float32} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vector{Float32}  # query (n_heads * head_size,)
    # k::Vector{Float32}   # key (dim,) - this is just a pointer to a portion of key_cache
    # v::Vector{Float32}   # value (dim,) - this is just a pointer to a portion of value_cache
    att::Array{Float32,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{Float32} # output logits (vocab_size,)

    # kv cache (n_kv_heads * head_size == kv_dim)
    key_cache::Array{Float32,3}  # (n_kv_heads * head_size, seq_len, n_layers)
    value_cache::Array{Float32,3}  # (n_kv_heads * head_size, seq_len, n_layers)
end

"""
A transformer model, consisting of a config, weights, and a run state.

llama2.c correspondence: Transformer (l. 67)
"""
struct Transformer
    config::Config # hyperparameters of the architecture
    weights::TransformerWeights # weights of the module
    state::RunState # buffers for the wave of activations in the forward pass
end

"""
    RunState(config::Config)

Initializes the matrices in RunState based on the shapes provided in the Config.

llama2.c correspondence: malloc_run_state (l. 77)
"""
function RunState(config::Config)
    # calculation of kv_dim
    kv_dim::Int32 = (config.dim * config.n_kv_heads) ÷ config.n_heads

    # initialization of undefined arrays
    return RunState(
        Array{Float32}(undef, config.dim),
        Array{Float32}(undef, config.dim),
        Array{Float32}(undef, config.dim),
        Array{Float32}(undef, config.hidden_dim),
        Array{Float32}(undef, config.hidden_dim),
        Array{Float32}(undef, config.dim),  # q
        # Array{Float32}(undef, config.dim),  # k
        # Array{Float32}(undef, config.dim),  # v
        Array{Float32}(undef, config.n_heads, config.seq_len),  # rms_att_weight
        Array{Float32}(undef, config.vocab_size),   # logits
        Array{Float32}(undef, kv_dim, config.seq_len, config.n_layers),
        Array{Float32}(undef, kv_dim, config.seq_len, config.n_layers),
    )
end

"""
    TransformerWeights(config::Config)

Initialize transformer weight matrices based on Config.

llama2.c correspondence: memory_map_weights (l. 111)
"""
function TransformerWeights(config::Config)
    (; dim, hidden_dim, n_heads, n_kv_heads, n_layers, vocab_size) = config
    kv_dim = (dim * n_kv_heads) ÷ n_heads

    # initialization of undefined arrays
    return TransformerWeights(
        Matrix{Float32}(undef, dim, vocab_size),
        Matrix{Float32}(undef, dim, n_layers),
        Matrix{Float32}(undef, dim, n_layers),
        Array{Float32}(undef, dim, dim, n_layers),
        Array{Float32}(undef, dim, kv_dim, n_layers),
        Array{Float32}(undef, dim, kv_dim, n_layers),
        Array{Float32}(undef, dim, dim, n_layers),
        Array{Float32}(undef, dim, hidden_dim, n_layers),
        Array{Float32}(undef, hidden_dim, dim, n_layers),
        Array{Float32}(undef, dim, hidden_dim, n_layers),
        Vector{Float32}(undef, config.dim),
        Matrix{Float32}(undef, dim, vocab_size),
    )
end

"""
    forward!(transformer::Transformer, token::Int, pos::Int)::Array{Float32}

A single complete transformer forward pass for input token `token` at position `pos`, returning the output logits.
`pos` is one-based, i.e. 1 <= pos <= seq_len.
This modifies the RunState of the transformer.

llama2.c correspondence: forward (l. 231)
"""
function forward!(transformer::Transformer, token::Int, pos::Int)::Array{Float32}
    # a few convenience variables
    (; dim, n_heads, n_kv_heads, n_layers, seq_len) = transformer.config
    1 <= pos <= seq_len || throw(ArgumentError("1 <= pos <= seq_len is required"))
    s = transformer.state
    w = transformer.weights
    kv_dim = (dim * n_kv_heads) ÷ n_heads
    kv_mul = n_heads ÷ n_kv_heads # integer multiplier of the kv sharing in multiquery
    head_size = dim ÷ n_heads

    # copy the token embedding into x
    s.x = w.token_embedding_table[:, token] # (dim,)

    # forward all the layers
    for l in 1:n_layers
        # attention rmsnorm
        #rmsnorm!(s.xb, s.x, w.rms_att_weight[:, l]) # (n_heads * head_size,)

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
            q = @view s.q[(h_off + 1):(h_off + head_size)] # (head_size,)
            # iterate over all timesteps, including the current one
            kv_ind = ((h - 1) ÷ kv_mul) * head_size
            for t in 1:pos
                k = @view s.key_cache[(kv_ind + 1):(kv_ind + head_size), t, l] # (head_size,)
                score = (q ⋅ k) / sqrt(Float32(head_size)) # scalar
                s.att[h, t] = score
            end
            # softmax the scores to get attention weights, from 1..pos inclusively
            softmax!(@view s.att[h, 1:pos]) # (pos,)
            # weighted sum of the values, store back into xb
            s.xb[(h_off + 1):(h_off + head_size)] .= 0 # (head_size,)
            for t in 1:pos
                # get the value vector for this head and at this timestep
                v = @view s.value_cache[(kv_ind + 1):(kv_ind + head_size), t, l] # (head_size,)
                # accumulate the weighted value into xb
                s.xb[(h_off + 1):(h_off + head_size)] += s.att[h, t] * v # (head_size,) = scalar * (head_size,)
            end
        end

        # final matmul to get the output of the attention
        mul!(s.xb2, w.wo[:, :, l]', s.xb) # (dim,) = (dim, n_heads * head_size) * (n_heads * head_size,)

        # residual connection back into x
        s.x += s.xb2 # (dim,) += (dim,)

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
        s.x += s.xb # (dim,)
    end

    # final rmsnorm
    rmsnorm!(s.x, s.x, w.rms_final_weight)
    # classifier into logits
    mul!(s.logits, w.wcls', s.x) # (vocab_size,) = (vocab_size, dim) * (dim,)
    return s.logits
end
