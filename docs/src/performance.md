```@meta
CurrentModule = Llama2
```

# Performance
The [`forward!`](@ref) method is optimized for quick execution in the following ways:
1. All operations (matrix multiplication, softmax, swiglu) are performed in-place. [[#35](https://github.com/kleincode/Llama2.jl/pull/35)]
2. All operations use views (using the `@views` macro). [[#35](https://github.com/kleincode/Llama2.jl/pull/35)]
3. The RoPE loop for calculating the positional embeddings supports multithreading using the `@threads` macro. [[#37](https://github.com/kleincode/Llama2.jl/pull/37)]

For benchmarks of these optimizations, check out the linked pull requests.