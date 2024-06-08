"""
Find the perfect match for a string in a given vocabulary

llama2.c correspondence: str_lookup (l. 445)
"""
# not needed
function str_lookup(char::Vector{String}, sorted_vocab::Dict{String,Int})
    id = get(sorted_vocab, char, -1)
    return id
end     # token_to_index[token]

"""
Encode the string text (input) into an upper-bound preallocated token array
bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)

llama2.c correspondence: encode (l. 452)
"""
function encode(tokenizer::Tokenizer, text::Vector{String})
    # temporary var that stores merge candidates of two consecuritve tokens
    tokens::Vector{Int32} = []
    str_buffer::String = ""    # contains all current tokens
    str_len::Int32 = 0

    # start at 0 tokens
    n_tokens::Array{Int32} = 0

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
            id::Int32 = tokenizer.token_to_index[merged_token]

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
        for j in (best_idx + 1):(length(tokens) - 1)
            tokens[j] = tokens[j + 1]
        end
    end

    # return encoded tokens
    return tokens
end