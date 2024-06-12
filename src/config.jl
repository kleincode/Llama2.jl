"""
Configuration of initial parameters

llama2.c correspondence Config (l.19)
"""
struct Config
    dim::Int32          # transformer dimension
    hidden_dim::Int32   # for ffn layers
    n_layers::Int32     # number of layers
    n_heads::Int32      # number of query heads
    n_kv_heads::Int32   # number of key/value heads (an be less than query heads)
    vocab_size::Int32   # vocabulary size
    seq_len::Int32      # max sequence length

    function Config(
        dim::Int32,
        hidden_dim::Int32,
        n_layers::Int32,
        n_heads::Int32,
        n_kv_heads::Int32,
        vocab_size::Int32,
        seq_len::Int32,
    )
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

        return new(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
    end
end