"""
Used to encode the tokenizer

llama.c correspondence: str_lookup (l. 445) and encode (l. 452)
"""

"""
Find the perfect match for a string in a given vocabulary
"""
function str_lookup(char::Vector{String}, sorted_vocab::Dict{String,Int})
    id = get(sorted_vocab, char, -1)
    return id
end

function enocode(tokenizer::Tokenizer) end