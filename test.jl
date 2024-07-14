using Llama2

config, weights = read_karpathy("bin/transformer/stories15M.bin") # replace with path to model weights
state = RunState{Float32}(config)
transformer = Transformer{Float32}(config, weights, state)
tokenizer = Tokenizer("bin/tokenizer/tokenizer.bin", config.vocab_size) # replace with path to tokenizer
sampler = Sampler{Float32}(1.0f0, 0.9f0, 420)

chat(transformer, tokenizer, sampler, cli_user_prompt = "Hello")