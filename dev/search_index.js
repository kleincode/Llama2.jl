var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Llama2","category":"page"},{"location":"#Llama2","page":"Home","title":"Llama2","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Llama2.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Llama2]","category":"page"},{"location":"#Llama2.Config","page":"Home","title":"Llama2.Config","text":"Configuration of initial parameters\n\nllama2.c correspondence Config (l.19)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.ProbIndex","page":"Home","title":"Llama2.ProbIndex","text":"Used when sorting probabilities during top-p sampling\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.RunState","page":"Home","title":"Llama2.RunState","text":"Initial parameters for the run state\n\nllama2.c correspondence: RunState (l. 50)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.RunState-Tuple{Config}","page":"Home","title":"Llama2.RunState","text":"Initialization of RunState based on Config\n\ninitialized as undefined arrays\n\nllama2.c correspondence: mallocrunstate (l. 77)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Sampler","page":"Home","title":"Llama2.Sampler","text":"Used to store sampling parameters.\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.Sampler-Tuple{Vector{Float32}}","page":"Home","title":"Llama2.Sampler","text":"(sampler::Sampler)(logits::Vector{Float32})\n\nSample the next token (id) based on the logits and the sampler parameters.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Sampler-Tuple{}","page":"Home","title":"Llama2.Sampler","text":"Sampler()\n\nCreate Sampler with default parameters\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Tokenizer","page":"Home","title":"Llama2.Tokenizer","text":"Used for mapping from strings to token arrays (Int vectors) and back.\n\nllama2.c correspondence: Tokenizer (l. 372)\n\nindextotoken = vocab\ntokentoindex = sorted_vocab\nremoved maxtokenlength (not required in Julia)\nremoved byte_pieces (not required in Julia)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.Tokenizer-Tuple{String, Int64}","page":"Home","title":"Llama2.Tokenizer","text":"Tokenizer(tokenizer_path::String, vocab_size::Int)\n\nConstructs a Tokenizer by loading the vocabulary from a file in the llama2.c format. The vocabulary size must be known from the config.\n\nllama2.c correspondence: build_tokenizer (l. 385)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Transformer","page":"Home","title":"Llama2.Transformer","text":"Initial parameters for the transformer\n\nllama2.c correspondence: Transformer (l. 67)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.TransformerWeights","page":"Home","title":"Llama2.TransformerWeights","text":"Initial parameters for the transformer weights\n\nllama2.c correspondence: TransformerWeights (l. 29)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.TransformerWeights-Tuple{Config}","page":"Home","title":"Llama2.TransformerWeights","text":"Initialize transformer weights based on Config\n\ninitialized as undefined arrays\n\nllama2.c correspondence: memorymapweights (l. 111)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.decode-Tuple{Tokenizer, Int64, Int64}","page":"Home","title":"Llama2.decode","text":"decode(tokenizer::Tokenizer, prev_token::Int32, token::Int32)\n\nDecodes a token index to a string. If the previous token is BOS, leading spaces are removed.\n\nllama2.c correspondence: decode (l. 418)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.rmsnorm-Tuple{Vector{Float32}, Vector{Float32}}","page":"Home","title":"Llama2.rmsnorm","text":"rmsnorm(x::Vector{Float32}, weight::Vector{Float32})\n\nCalculate the root mean square norm of a vector.  Reference in llama2.c lines 182-195\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.sample_argmax-Tuple{Vector{Float32}}","page":"Home","title":"Llama2.sample_argmax","text":"sample_argmax(logits::Vector{Float32})\n\nReturn the index that has the highest probability\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.sample_mult-Tuple{Vector{Float32}, Float32}","page":"Home","title":"Llama2.sample_mult","text":"sample_mult(probabilities::Vector{Float32}, coin::Float32)\n\nSample index from probabilities (they must sum to 1!). Coin is a random number in [0, 1). Find the index that coin falls into.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.sample_topp-Tuple{Vector{Float32}, Float32, Float32}","page":"Home","title":"Llama2.sample_topp","text":"sample_topp(probabilities::Vector{Float32}, topp::Float32, coin::Float32)\n\nTop-p sampling (or \"nucleus sampling\") samples from the smallest set of tokens that exceed probability topp. This way we never sample tokens that have very low probabilities and are less likely to go \"off the rails\". Coin is a random number in [0, 1)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.softmax-Tuple{Vector{Float32}}","page":"Home","title":"Llama2.softmax","text":"softmax(x::Vector{Float32})\n\nCalculate the softmax of a vector.  Reference in llama2.c lines 197-215\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.swiglu-Tuple{Vector{Float32}, Vector{Float32}}","page":"Home","title":"Llama2.swiglu","text":"swiglu(x::Vector{Float32}, x2::Vector{Float32})\n\nActivation function that combines GLU and Swish functions.  Reference in llama2.c lines 338-345\n\n\n\n\n\n","category":"method"}]
}