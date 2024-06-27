"""
    read_karpathy(file_path::String)
Reads Config and TransformerWeights from a Karpathy binary file, as defined in llama2.c.
"""
function read_karpathy(file_path::String)
    open(file_path, "r") do file
        config = read_karpathy_config(file)
        weights = read_karpathy_weights(config, file)
        return config, weights
    end
end

"""
    read_karpathy_weights(config::Config, file::IOStream)
Reads TransformerWeights from a Karpathy binary file, as defined in llama2.c.
The function assumes that the provided `IOStream` is already pointing to the beginning of the weights section.

The classifier weights are always assumed to be identical to the `token_embedding_table`
because the config is guaranteed to have positive `vocab_size`, meaning that the weights are shared.

llama2.c correspondence: memory_map_weights (l. 111)
"""
function read_karpathy_weights(config::Config, file::IOStream)
    weights = TransformerWeights(config)
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
    read_karpathy_config(file::IOStream)
Reads a Config instance from a Karpathy binary file, as defined in llama2.c

Note that this function advances the pointer of file by 7 Int32s.
The config section of a Karpathy binary file is by definition the beginning of the file.

llama2.c correspondence: read_config (l. 147)
"""

function read_karpathy_config(file::IOStream)
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
