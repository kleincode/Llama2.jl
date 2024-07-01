```@meta
CurrentModule = Llama2
```

# Getting Started
The model weights can be downloaded from Andrew Karpathy's HuggingFace [tinyllamas repo](https://huggingface.co/karpathy/tinyllamas/tree/main). The default tokenizer, which is compatible with `stories110M.bin`, `stories15M.bin`, and `stories42M.bin`, can be downloaded from [our repository](https://github.com/kleincode/Llama2.jl/tree/main/bin/tokenizer) or from the [llama2.c repository](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin).

Add the package to your local environment via Pkg by running
```bash
add https://github.com/kleincode/Llama2.jl
```

Import the package using
```julia-repl
julia> using Llama2
```

## Generate text

```julia
config, weights = read_karpathy("bin/transformer/stories15M.bin")
state = RunState{Float32}(config)
transformer = Transformer{Float32}(config, weights, state)
tokenizer = Tokenizer("bin/tokenizer/tokenizer.bin", 32000)
sampler = Sampler(1.0f0, 0.9f0, 420)

prompt = "Once upon a"
generate(transformer, tokenizer, sampler, prompt, false)
```
```
"Once upon a time, there was a little girl named Lily. She loved to help her mommy in the kitchen. One day, her mommy was making some yummy cookies and asked Lily to help her. Lily was so excited! \nShe put on her apron and stood on a stool so she could reach the cookies. Her mommy was so proud of her independent little girl. They mixed the ingredients together and put the cookies in the oven. \nAfter a little while, the cookies were ready and they smelled delicious. Lily's mommy let her have a slice and they both enjoyed the warm, tasty cookie. From that day on, Lily loved to help her mommy in the kitchen and help cook yummy treats."
```

## Use the tokenizer
```julia-repl
julia> tokenizer = Tokenizer("bin/tokenizer/tokenizer.bin", 32000)
Tokenizer{Float32}(["<unk>", "\n<s>\n", "\n</s>\n", "<0x00>", "<0x01>", "<0x02>", "<0x03>", "<0x04>", "<0x05>", "<0x06>"  …  "ὀ", "げ", "べ", "边", "还", "黃", "왕", "收", "弘", "给"], Dict("âr" => 28727, " properly" => 6285, "chem" => 14970, " patients" => 22070, " Plan" => 8403, "<0x2A>" => 46, "рос" => 10375, "null" => 4305, "rę" => 15387, "ört" => 21069…), Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  -31731.0, -31732.0, -31733.0, -31734.0, -31735.0, -31736.0, -31737.0, -31738.0, -31739.0, -31740.0])
julia> encode(tokenizer, "Tokens are beautiful! :))")
8-element Vector{Int64}:
     2
 11891
   576
   527
  9561
 29992
   585
   877
julia> decode(tokenizer, 2, 11891)
"Tok"
```
For more information, check out [`Tokenizer`](@ref).