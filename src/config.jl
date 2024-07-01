"""
$(TYPEDEF)

Used to configure the initial parameters.

## Fields
$(TYPEDFIELDS)

Initializes parameters and checks for the correct dimensions. 
For example, the config can be read from a file using the [`read_karpathy_config`](@ref) function and is part of the [`TransformerWeights`](@ref) function. 

llama2.c correspondence Config (l.19)
"""
struct Config{T<:Integer}
    """Transformer Dimension"""
    dim::T  
    """ffn Layers"""        
    hidden_dim::T   
    """Number of Layers"""
    n_layers::T     
    """Number of Query Heads"""
    n_heads::T      
    """Number of key/value heads"""
    n_kv_heads::T   # can be less than query heads
    """Vocabulary Size"""
    vocab_size::T  
    """Max Sequence Length"""
    seq_len::T     

    function Config{T}(
        dim::T,
        hidden_dim::T,
        n_layers::T,
        n_heads::T,
        n_kv_heads::T,
        vocab_size::T,
        seq_len::T,
    ) where {T<:Integer}
        # Input checks
        dim > 0 || throw(ArgumentError("Dimension must be bigger than zero"))
        hidden_dim > 0 || throw(ArgumentError("Hidden dimension must be bigger than zero"))
        n_layers > 0 ||
            throw(ArgumentError("The number of layers must be bigger than zero"))
        n_heads > 0 ||
            throw(ArgumentError("The number of query heads must be bigger than zero"))
        n_kv_heads > 0 ||
            throw(ArgumentError("The number of key/value heads must be bigger than zero"))
        vocab_size > 0 || throw(ArgumentError("The vocabulary must not be empty"))
        seq_len > 0 || throw(ArgumentError("The sequence length must be bigger than zero"))
        dim % n_heads == 0 || throw(ArgumentError("dim must be a multiple of n_heads"))
        n_heads % n_kv_heads == 0 ||
            throw(ArgumentError("n_heads must be a multiple of n_kv_heads"))

        return new{T}(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
    end
end