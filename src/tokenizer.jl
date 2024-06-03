"""
Used for mapping from strings to token arrays (Int vectors) and back.

llama.c correspondence: Tokenizer
- index_to_token = vocab
- token_to_index = sorted_vocab
- removed max_token_length (not required in Julia)
- removed byte_pieces (not required in Julia)
"""
struct Tokenizer
    "Maps a token index to its string representation, for decoding"
    index_to_token::Vector{String}
    "Maps a token string to its token index, for encoding"
    token_to_index::Dict{String,Int} # for encoding
    "Scores of individual tokens for encoding"
    vocab_scores::Vector{Float32} # for encoding
end

"""
    Tokenizer(tokenizer_path::String, vocab_size::Int)

Constructs a Tokenizer by loading the vocabulary from a file in the llama2.c format.
The vocabulary size must be known from the config.
"""
function Tokenizer(tokenizer_path::String, vocab_size::Int)
    index_to_token = Vector{String}(undef, vocab_size)
    token_to_index = Dict{String,Int}()
    vocab_scores = Vector{Float32}(undef, vocab_size)
    # Read the file
    open(tokenizer_path) do f
        read(f, Int32) # max_token_length, ignored
        for i in 1:vocab_size
            vocab_scores[i] = read(f, Float32)
            token_length = read(f, Int32)
            token = String(read(f, token_length))
            index_to_token[i] = token
            token_to_index[token] = i
        end
    end
    return Tokenizer(index_to_token, token_to_index, vocab_scores)
end