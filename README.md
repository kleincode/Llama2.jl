# Llama2.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kleincode.github.io/Llama2.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kleincode.github.io/Llama2.jl/dev/)
[![Build Status](https://github.com/kleincode/Llama2.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kleincode/Llama2.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kleincode/Llama2.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kleincode/Llama2.jl)

This is a port of Andrew Karpathy's [llama2.c](https://github.com/karpathy/llama2.c/) to Julia.

> [!IMPORTANT]  
> This is part of the JuliaML course at Technical University Berlin. Therefore, the repository will not be maintained for a long time.

## Features
- Read tokenizer and model weights from `.bin` files specified in llama2.c
- Inference, generation loop and chat loop
- argmax, multinomial and top-p sampling
- Tokenizer for encoding text to LLM input and decoding LLM output to text

## Getting started
Add the package to your local environment via Pkg by running
```bash
add https://github.com/kleincode/Llama2.jl
```
To get started, check out the [docs](https://kleincode.github.io/Llama2.jl).