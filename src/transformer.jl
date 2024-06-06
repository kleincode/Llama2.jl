"""
Initial parameters for the transformer weights

llama.c correspondence: TrasformerWeights (l. 29)
"""
struct TransformerWeights    # using @kwdef for clearer parameter initialization in the outer constructor 
    # token embedding table
    token_embedding_table::Matrix{Float32} # (vocab_size, dim)

    # weights
    rms_att_weight::Matrix{Float32}         # (layer, dim)
    rms_ffn_weight::Matrix{Float32}         # (layer, dim)

    # weights for matmuls (dim == n_heads * head_size)
    wq::Array{Float32,3}  # (layer, dim, n_heads * head_size)
    wk::Array{Float32,3}  # (layer, dim, n_kv_heads * head_size)
    wv::Array{Float32,3}  # (layer, dim, n_kv_heads * head_size)
    wo::Array{Float32,3}  # (layer, n_heads * head_size, dim)

    # weights for ffn
    w1::Array{Float32,3}  # (layer, hidden_dim, dim)
    w2::Array{Float32,3}  # (layer, dim, hidden_dim)
    w3::Array{Float32,3}  # (layer, hidden_dim, dim)

    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)

    # optional classifier weights
    # wcls::Array{Float32} 
end

"""
Initial parameters for the run state

llama.c correspondence: RunState (l. 50)
"""
@kwdef struct RunState  # using @kwdef for clearer parameter initialization in the outer constructor
    # current way of activations
    x::Vector{Float32}  # activation at current time stamp (dim,)
    xb::Vector{Float32}  # same, but inside a residual branch (dim,)
    xb2::Vector{Float32}  # an additional buffer just for convenience (dim,)
    hb::Vector{Float32}  # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vector{Float32} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vector{Float32}  # query (dim,)
    k::Vector{Float32}   # key (dim,)
    v::Vector{Float32}   # value (dim,)
    att::Array{Float32,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{Float32} # output logits

    # kv cache
    key_cache::Array{Float32,3}  # (layer, seq_len, dim)
    value_cache::Array{Float32,3}  # (layer, seq_len, dim)
end

"""
Initial parameters for the transformer

llama.c correspondence: Transformer (l. 67)
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

llama.c correspondence: malloc_run_state (l. 77)
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
        Array{Float32}(undef, config.dim),
        Array{Float32}(undef, config.n_layers, config.seq_len, kv_dim),
        Array{Float32}(undef, config.n_layers, config.seq_len, kv_dim),
        Array{Float32}(undef, config.seq_len, config.n_heads),
        Array{Float32}(undef, config.vocab_size),
    )
end

"""
Initialize transformer weights based on Config
- intitialized as undefined arrays

llama.c correspondence: memory_map_weights (l. 111)
"""
function TransformerWeights(config::Config)
    # calculation of head_size
    head_size::Int32 = config.dim ÷ config.n_heads

    # initialization of undefined arrays
    return TransformerWeights(
        Matrix{Float32}(undef, config.vocab_size, config.dim),
        Matrix{Float32}(undef, config.n_layers, config.dim),
        Matrix{Float32}(undef, config.n_layers, config.dim),
        Array{Float32}(undef, config.n_layers, config.dim, (config.n_heads * head_size)),
        Array{Float32}(undef, config.n_layers, config.dim, (config.n_kv_heads * head_size)),
        Array{Float32}(undef, config.n_layers, config.dim, (config.n_kv_heads * head_size)),
        Array{Float32}(undef, config.n_layers, (config.n_heads * head_size), config.dim),
        Array{Float32}(undef, config.n_layers, config.hidden_dim, config.dim),
        Array{Float32}(undef, config.n_layers, config.dim, config.hidden_dim),
        Array{Float32}(undef, config.n_layers, config.hidden_dim, config.dim),
        Vector{Float32}(undef, config.dim),
    )
end