"""
$(TYPEDEF)

Used for mapping from strings to token arrays (Int vectors) and back.

## Fields
$(TYPEDFIELDS)

llama2.c correspondence: Tokenizer (l. 372)
- index_to_token = vocab
- token_to_index = sorted_vocab
- removed max_token_length (not required in Julia)
- removed byte_pieces (not required in Julia)

# Load from Karpathy bin file
    Tokenizer(tokenizer_path::String, vocab_size::Int)

Constructs a Tokenizer by loading the vocabulary from a file in the llama2.c format.
The vocabulary size must be known from the config.

## Example
```julia-repl
julia> Tokenizer("bin/tokenizer/tokenizer.bin", 32000)
Tokenizer(["<unk>", "\n<s>\n", "\n</s>\n", "<0x00>", "<0x01>", "<0x02>", "<0x03>", "<0x04>", "<0x05>", "<0x06>"  …  "ὀ", "げ", "べ", "边", "还", "黃", "왕", "收", "弘", "给"], Dict("âr" => 28727, " properly" => 6285, "chem" => 14970, " patients" => 22070, " Plan" => 8403, "<0x2A>" => 46, "рос" => 10375, "null" => 4305, "rę" => 15387, "ört" => 21069…), Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  -31731.0, -31732.0, -31733.0, -31734.0, -31735.0, -31736.0, -31737.0, -31738.0, -31739.0, -31740.0])
```

llama2.c correspondence: build_tokenizer (l. 385)
"""
struct Tokenizer{T<:Real}
    "Maps a token index to its string representation, for decoding"
    index_to_token::Vector{String}
    "Maps a token string to its token index, for encoding"
    token_to_index::Dict{String,Int} # for encoding
    "Scores of individual tokens for encoding"
    vocab_scores::Vector{T} # for encoding

    "Constructs a Tokenizer from a list of tokens and scores."
    function Tokenizer(tokens::Vector{String}, scores::Vector{T}) where {T<:Real}
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
        return new{T}(tokens, token_to_index, scores)
    end
end

function Tokenizer(tokenizer_path::String, vocab_size::Integer)
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

const BOS_TOKEN::Int32 = 2
const EOS_TOKEN::Int32 = 3

"""
$(TYPEDSIGNATURES)

Encode a string text using a [`Tokenizer`](@ref). 
An optional EOS token can be added.
Encoded text can be decoded with the [`decode`](@ref) function.

Works by encoding each code unit as a single token, then iteratively merging them together according to the [`Tokenizer`](@ref)'s `vocab_scores`.

Note that token indices are 1-based (different to the 0-based system in the llama2.c).

## Example
```julia-repl
julia> encode(tokenizer, "Hello world!")
4-element Vector{Int64}:
     2
 15044
  3187
 29992
```

llama2.c correspondence: encode (l. 452)
"""
function encode(tokenizer::Tokenizer, text::String, eos_token::Bool=false)
    # stores merge candidates of two consecutive tokens
    tokens::Vector{Int} = []

    # add BOS token
    push!(tokens, BOS_TOKEN)
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
        if haskey(tokenizer.token_to_index, token)
            push!(tokens, tokenizer.token_to_index[token])
        else
            representation = "<0x" * uppercase(string(c, base=16, pad=2)) * ">"
            if haskey(tokenizer.token_to_index, representation)
                push!(tokens, tokenizer.token_to_index[representation])
            else
                throw(KeyError(token))
            end
        end
    end

    # merge the best consecutive pair each iteration
    while true
        best_score::Real = -1.0f10
        best_id::Int32 = -1
        best_idx::Int32 = -1

        for i in 1:(length(tokens) - 1)
            # merge consecutive tokens 
            merged_token =
                tokenizer.index_to_token[tokens[i]] *
                tokenizer.index_to_token[tokens[i + 1]]

            # check if merged_token exists
            id = get(tokenizer.token_to_index, merged_token, nothing)

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

    # add EOS token, optional
    if eos_token == true
        push!(tokens, EOS_TOKEN)
    end

    # return encoded tokens
    return tokens
end

"""
$(TYPEDSIGNATURES)

Decodes a token index to a string.
If the previous token is BOS (=2) and the token value starts with a leading space, the leading space is removed.
Token indices are 1-based (different to the 0-based system in llama2.c).

## Example
```julia-repl
julia> [decode(tokenizer, 1, t) for t in [2, 15044, 3187, 29992]]
4-element Vector{String}:
 "\n<s>\n"
 " Hello"
 " world"
 "!"

julia> decode(tokenizer, 1, 15044)
" Hello"

julia> decode(tokenizer, 2, 15044) # BOS strips leading space
"Hello"
```

llama2.c correspondence: decode (l. 418)
"""
function decode(tokenizer::Tokenizer, prev_token::Int, token::Int)
    piece = tokenizer.index_to_token[token]
    # following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if prev_token == BOS_TOKEN && piece[1] == ' '
        piece = piece[2:end]
    end
    # careful, some tokens esignate raw bytes, and look like e.g. '<0x01>'
    # parse this and convert and return the actual byte
    if startswith(piece, "<0x") && piece[end] == '>'
        return string(Char(parse(Int, piece[4:(end - 1)]; base=16)))
    else
        return piece
    end
end
