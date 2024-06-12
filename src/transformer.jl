using LinearAlgebra

"""
Initial parameters for the transformer weights

llama2.c correspondence: TransformerWeights (l. 29)
"""
struct TransformerWeights
    # token embedding table
    token_embedding_table::Matrix{Float32} # (vocab_size, dim)

    # weights
    rms_att_weight::Matrix{Float32}         # (layer, dim)
    rms_ffn_weight::Matrix{Float32}         # (layer, dim)

    # weights for matmuls (dim == n_heads * head_size)
    wq::Array{Float32,3}  # (n_layers, dim, dim)
    wk::Array{Float32,3}  # llama2.c says (n_layers, dim, kv_dim), should be (n_layers, kv_dim, dim)
    wv::Array{Float32,3}  # llama2.c says (n_layers, dim, kv_dim), should be (n_layers, kv_dim, dim)
    wo::Array{Float32,3}  # (n_layers, dim, dim)

    # weights for ffn
    w1::Array{Float32,3}  # (layer, hidden_dim, dim)
    w2::Array{Float32,3}  # (layer, dim, hidden_dim)
    w3::Array{Float32,3}  # (layer, hidden_dim, dim)

    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)

    # optional classifier weights
    wcls::Matrix{Float32}  # (vocab_size, dim)
end

"""
Initial parameters for the run state

llama2.c correspondence: RunState (l. 50)
"""
mutable struct RunState
    # current way of activations
    x::Vector{Float32}  # activation at current time stamp (dim,)
    xb::Vector{Float32}  # same, but inside a residual branch (dim,)
    xb2::Vector{Float32}  # an additional buffer just for convenience (dim,)
    hb::Vector{Float32}  # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vector{Float32} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vector{Float32}  # query (dim,)
    # k::Vector{Float32}   # key (dim,) - this is just a pointer to a portion of key_cache
    # v::Vector{Float32}   # value (dim,) - this is just a pointer to a portion of value_cache
    att::Array{Float32,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{Float32} # output logits (vocab_size,)

    # kv cache
    key_cache::Array{Float32,3}  # (n_layers, seq_len, kv_dim)
    value_cache::Array{Float32,3}  # (n_layers, seq_len, kv_dim)
end

"""
Initial parameters for the transformer

llama2.c correspondence: Transformer (l. 67)
"""
struct Transformer
    config::Config # hyperparameters of the architecture
    weights::TransformerWeights # weights of the module
    state::RunState # buffers for the wave of activations in the forward pass

    # some more states for clean up of memory mapping (do we need that?)
    fd::Int32 # file descriptor for memory mapping
    data::Array{Float32} # memory mapped data pointer
    file_size::Int32 # size of checkpoint file in bytes
end

"""
Initialization of RunState based on Config
- initialized as undefined arrays

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
        Array{Float32}(undef, config.n_layers, config.seq_len, kv_dim),
        Array{Float32}(undef, config.n_layers, config.seq_len, kv_dim),
    )
end

"""
Initialize transformer weights based on Config
- initialized as undefined arrays

llama2.c correspondence: memory_map_weights (l. 111)
"""
function TransformerWeights(config::Config)
    (; dim, hidden_dim, n_heads, n_kv_heads, n_layers, vocab_size) = config
    kv_dim = (dim * n_kv_heads) ÷ n_heads

    # initialization of undefined arrays
    return TransformerWeights(
        Matrix{Float32}(undef, vocab_size, dim),
        Matrix{Float32}(undef, n_layers, dim),
        Matrix{Float32}(undef, n_layers, dim),
        Array{Float32}(undef, n_layers, dim, dim),
        Array{Float32}(undef, n_layers, kv_dim, dim),
        Array{Float32}(undef, n_layers, kv_dim, dim),
        Array{Float32}(undef, n_layers, dim, dim),
        Array{Float32}(undef, n_layers, hidden_dim, dim),
        Array{Float32}(undef, n_layers, dim, hidden_dim),
        Array{Float32}(undef, n_layers, hidden_dim, dim),
        Vector{Float32}(undef, dim),
        Matrix{Float32}(undef, vocab_size, dim),
    )
end

function forward(transformer::Transformer, token::Int, pos::Int)::Array{Float32}
    # a few convenience variables
    (; dim, n_heads, n_kv_heads, n_layers) = transformer.config
    s = transformer.state
    w = transformer.weights
    kv_dim = (dim * n_kv_heads) ÷ n_heads
    kv_mul = n_heads ÷ n_kv_heads # integer multiplier of the kv sharing in multiquery
    head_size = dim ÷ n_heads

    # copy the token embedding into x
    x::Vector{Float32} = transformer.weights.token_embedding_table[token, :]

    # forward all the layers
    for l in 1:n_layers
        # attention rmsnorm
        s.xb = rmsnorm(x, w.rms_att_weight[l, :])

        # qkv matmuls for this position
        s.q = w.wq[l, :, :] * s.xb # (dim,) = (dim,dim) * (dim,)
        s.key_cache[l, pos, :] = w.wk[l, :, :] * s.xb # (kv_dim,) = (kv_dim,dim) * (dim,)
        s.value_cache[l, pos, :] = w.wv[l, :, :] * s.xb # (kv_dim,) = (kv_dim,dim) * (dim,)

        # RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in 1:2:dim
            head_dim = (i - 1) % head_size
            freq = 1.0f0 / (10000.0f0^(head_dim / head_size))
            val = pos * freq
            fcr = cos(val)
            fci = sin(val)

            # rotate query
            v0 = s.q[i]
            v1 = s.q[i + 1]
            s.q[i] = v0 * fcr - v1 * fci
            s.q[i + 1] = v0 * fci + v1 * fcr

            # rotate key only if i < kv_dim
            if i < kv_dim
                v0 = s.key_cache[l, pos, i]
                v1 = s.key_cache[l, pos, i + 1]
                s.key_cache[l, pos, i] = v0 * fcr - v1 * fci
                s.key_cache[l, pos, i + 1] = v0 * fci + v1 * fcr
            end
        end

        # iterate over all heads
        for h in 0:(n_heads - 1)
            # get the query vector for this head
            h_off = h * head_size
            q = s.q[(h_off + 1):(h_off + head_size)]
            # iterate over all timesteps, including the current one
            kv_ind = (h ÷ kv_mul) * head_size
            for t in 1:pos
                k = s.key_cache[l, t, (kv_ind + 1):(kv_ind + head_size)]
                score = (q ⋅ k) / sqrt(Float32(head_size))
                s.att[h + 1, t] = score
            end
            # softmax the scores to get attention weights, from 1..pos inclusively
            s.att[h + 1, 1:pos] = softmax(s.att[h + 1, 1:pos])
            # weighted sum of the values, store back into xb
            s.xb[(h_off + 1):(h_off + head_size)] .= 0
            for t in 1:pos
                # get the value vector for this head and at this timestep
                v = s.value_cache[l, t, (kv_ind + 1):(kv_ind + head_size)]
                # accumulate the weighted value into xb
                s.xb[(h_off + 1):(h_off + head_size)] += s.att[h + 1, t] * v
            end
        end

        # final matmul to get the output of the attention
        s.xb2 = w.wo[l, :, :] * s.xb

        # residual connection back into x
        x += s.xb2

        # ffn rmsnorm
        s.xb = rmsnorm(s.xb, w.rms_ffn_weight[l, :])

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        s.hb = w.w1[l, :, :] * s.xb
        s.hb2 = w.w3[l, :, :] * s.xb

        # SwiGLU non-linearity
        s.hb = swiglu(s.hb, s.hb2)
        # final matmul to get the output of the ffn
        s.xb = w.w2[l, :, :] * s.hb
        # residual connection
        x += s.xb
    end

    # final rmsnorm
    x = rmsnorm(x, w.rms_final_weight)
    # classifier into logits
    s.logits = w.wcls * x
    return s.logits
end