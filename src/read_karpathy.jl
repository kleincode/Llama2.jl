"""
$(TYPEDSIGNATURES)

Reads a Karpathy file and returns the Config and Weights using the [`read_karpathy_config`](@ref) function
and the [`read_karpathy_weights`](@ref) function.
"""
function read_karpathy(file_path::String)
    open(file_path, "r") do file
        config = read_karpathy_config(file)
        weights = read_karpathy_weights(config, file)
        return config, weights
    end
end

"""
$(TYPEDSIGNATURES)

Read the weights of a Karpathy file and return them using the [`TransformerWeights`](@ref) function.

llama2.c correspondence: memory_map_weights (l. 111)
"""
function read_karpathy_weights(config::Config, file::IOStream)
    weights = TransformerWeights{Float32}(config)
    # read weights from file
    read!(file, weights.token_embedding_table)
    read!(file, weights.rms_att_weight)
    read!(file, weights.wq)
    read!(file, weights.wk)
    read!(file, weights.wv)
    read!(file, weights.wo)
    read!(file, weights.rms_ffn_weight)
    read!(file, weights.w1)
    read!(file, weights.w2)
    read!(file, weights.w3)
    read!(file, weights.rms_final_weight)

    # optional classifier weights
    # read!(file, weights.wcls)
    # usually identical -> just copy
    weights.wcls[:] = weights.token_embedding_table[:]
    return weights
end

"""
$(TYPEDSIGNATURES)

Read the config of a Karpathy file and return that Configuration using the [`config`](@ref) function. 

llama2.c correspondence: read_config (l. 147)
"""
function read_karpathy_config(file::IOStream)
    # read config from file
    return Config{Int32}(
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
        read(file, Int32),
    )
end
