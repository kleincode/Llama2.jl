"""
Initial parameters for the transformer weights

llama2.c correspondence: TransformerWeights (l. 29)
"""
struct TransformerWeights
    # token embedding table
    token_embedding_table::Matrix{Float32} # (dim, vocab_size)

    # weights
    rms_att_weight::Matrix{Float32}         # (dim, layer)
    rms_ffn_weight::Matrix{Float32}         # (dim, layer)

    # weights for matmuls (dim == n_heads * head_size)
    wq::Array{Float32,3}  # (n_heads * head_size, layer, dim)
    wk::Array{Float32,3}  # (n_kv_heads * head_size, layer, dim)
    wv::Array{Float32,3}  # (n_kv_heads * head_size, layer, dim)
    wo::Array{Float32,3}  # (n_kv_heads * head_size, layer, dim)

    # weights for ffn
    w1::Array{Float32,3}  # (dim, layer, hidden_dim)
    w2::Array{Float32,3}  # (hidden_dim, dim, layer)
    w3::Array{Float32,3}  # (dim, hidden_dim, layer)

    # final rmsnorm
    rms_final_weight::Vector{Float32} # (dim,)

    # optional classifier weights
    # wcls::Array{Float32} 
end

"""
Initial parameters for the run state

llama2.c correspondence: RunState (l. 50)
"""
struct RunState
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
    logits::Array{Float32} # output logits, not sure which dimensions

    # kv cache
    key_cache::Array{Float32,3}  # (layer, seq_len, dim)
    value_cache::Array{Float32,3}  # (layer, seq_len, dim)
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
        # in llama2.c k and v are not defined
        Array{Float32}(undef, config.dim),  # k
        Array{Float32}(undef, config.dim),  # v
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
    # calculation of head_size
    head_size::Int32 = config.dim ÷ config.n_heads

    # initialization of undefined arrays
    return TransformerWeights(
        Matrix{Float32}(undef, config.dim, config.vocab_size),
        Matrix{Float32}(undef, config.dim, config.n_layers),
        Matrix{Float32}(undef, config.dim, config.n_layers),
        Array{Float32}(undef, (config.n_heads * head_size), config.dim, config.n_layers,),
        Array{Float32}(undef, (config.n_kv_heads * head_size), config.dim, config.n_layers,),
        Array{Float32}(undef, (config.n_kv_heads * head_size), config.dim, config.n_layers,),
        Array{Float32}(undef, config.dim, (config.n_heads * head_size), config.n_layers,),
        Array{Float32}(undef, config.dim, config.hidden_dim, config.n_layers),
        Array{Float32}(undef, config.hidden_dim, config.dim, config.n_layers),
        Array{Float32}(undef, config.dim, config.hidden_dim, config.n_layers,),
        Vector{Float32}(undef, config.dim),
    )
end

"""
TODO - what does this function do?
"""
function open_file(file_path::String)
    file = open(file_path, "r")
    config = read_config(file)
    weights = readLlamaFiles(config, file)
    return config, weights
end



function readLlamaFiles(config::Config, file::IOStream)
    weights = TransformerWeights(config)
    # read weights from file
    read!(file, weights.token_embedding_table)
    read!(file, weights.rms_att_weight)
    read!(file, weights.rms_ffn_weight)
    read!(file, weights.wq)
    read!(file, weights.wk)
    read!(file, weights.wv)
    read!(file, weights.wo)
    read!(file, weights.w1)
    read!(file, weights.w2)
    read!(file, weights.w3)
    read!(file, weights.rms_final_weight)

    # optional classifier weights
    # read!(file, weights.wcls)
    return weights
end

"""
Read the config from a Kaparthy file

llama2.c correspondence: read_config (l. 147)
"""

function read_config(file::IOStream)
    # read config from file
    config = Config(
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
    )
    return config
end