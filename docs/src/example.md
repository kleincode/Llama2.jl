```@meta
CurrentModule = Llama2
```

# Getting Started
The model weights can be downloaded from Andrew Karpathy's HuggingFace [tinyllamas repo](https://huggingface.co/karpathy/tinyllamas/tree/main). The default tokenizer, which is compatible with `stories110M.bin`, `stories15M.bin`, and `stories42M.bin`, can be downloaded from [our repository](https://github.com/kleincode/Llama2.jl/tree/main/bin/tokenizer) or from the [llama2.c repository](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin).

Add the package to your local environment via Pkg by running
```bash
add https://github.com/kleincode/Llama2.jl
```

## Generate text

```julia
using Llama2

config, weights = read_karpathy("../bin/transformer/stories15M.bin")
state = RunState(config)
transformer = Transformer(config, weights, state)
tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", 32000)
sampler = Sampler(1.0f0, 0.9f0, 420)

prompt = "Once upon a"
generate(transformer, tokenizer, sampler, prompt, false)
```
```
Once upon a time...
```