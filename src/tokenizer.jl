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

"""
Encode the string text (input) into an upper-bound preallocated token array
bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)

llama2.c correspondence: encode (l. 452)
"""
function encode(tokenizer::Tokenizer, text::Vector{String})
    # temporary var that stores merge candidates of two consecuritve tokens
    tokens::Vector{Int32} = []

    # if the text is empty there is no encoding so we can return the empty tokens array
    if isempty(text)
        return tokens
    end
    # add dummy_prefix is default
    # add prefix to the input string only if text isn't empty
    # not sure why though
    if text != ""
        dummy_prefix::Int32 = tokenizer.token_to_index[" "]
        push!(tokens, dummy_prefix)
    end

    # process the bytes of the text
    # no need for str_buffers (as in llama2.c) because of julias native codeunits
    for c in codeunits(text)
        # add token of the byte to the tokens list
        token = String(UInt8[c])
        push!(tokens, tokenizer.token_to_index[token])
    end

    # merge the best consecutive pair each iteration
    while true
        best_score::Float32 = -1e10
        best_id::Int32 = -1
        best_idx::Int32 = -1

        for i in 1:(length(tokens) - 1)
            # merge consecutive tokens 
            merged_token =
                tokenizer.index_to_token[tokens[i]] *
                tokenizer.index_to_token[tokens[i + 1]]

            # check if merged_token exists
            id::Int32 = get(tokenizer.token_to_index, merged_token, nothing)

            # if id exists and the vocab_score is bigger than the best score, this token becomes the new best token
            if !isnothing(id) && tokenizer.vocab_scores[id] > best_score
                best_score = tokenizer.vocab_scores[id]
                best_id = id
                best_idx = i
            end
        end

        # if no more tokens can be merged, we break
        if best_idx == -1
            break
        end

        # merge consecutive pair into new token best_id
        tokens[best_idx] = best_id

        # delete token at position best_idx+1, shift entire sequence back 1
        splice!(tokens, best_idx + 1)
    end

    # return encoded tokens
    return tokens
end