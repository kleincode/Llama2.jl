"""
Used for mapping from strings to token arrays (Int vectors) and back.

llama.c correspondence: Tokenizer (l. 372)
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

    "Constructs a Tokenizer from a list of tokens and scores."
    function Tokenizer(tokens::Vector{String}, scores::Vector{Float32})
        # Input checks
        n = length(tokens)
        n > 0 || throw(ArgumentError("Tokens must not be empty"))
        length(scores) == n ||
            throw(ArgumentError("Tokens and scores must have the same length"))
        all(token -> token != "", tokens) ||
            throw(ArgumentError("Tokens must not contain empty strings"))

        # Construct the reverse mapping
        token_to_index = Dict{String,Int}()
        for (index, token) in pairs(tokens)
            if haskey(token_to_index, token)
                throw(ArgumentError("Duplicate token: $token"))
            end
            token_to_index[token] = index
        end
        return new(tokens, token_to_index, scores)
    end
end

"""
    Tokenizer(tokenizer_path::String, vocab_size::Int)

Constructs a Tokenizer by loading the vocabulary from a file in the llama2.c format.
The vocabulary size must be known from the config.

llama.c correspondence: build_tokenizer (l. 385)
"""
function Tokenizer(tokenizer_path::String, vocab_size::Int)
    tokens = Vector{String}(undef, vocab_size)
    vocab_scores = Vector{Float32}(undef, vocab_size)
    # Read the file
    open(tokenizer_path) do f
        read(f, Int32) # max_token_length, ignored
        for i in 1:vocab_size
            # file format: score, k, t_1, t_2, ..., t_k
            vocab_scores[i] = read(f, Float32)
            token_length = read(f, Int32)
            tokens[i] = String(read(f, token_length))
        end
    end
    return Tokenizer(tokens, vocab_scores)
end