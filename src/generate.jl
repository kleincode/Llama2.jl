"""
Used to generate a sequence based on a given language model.


llama.c correspondence: generation loop (l. 729-783)
"""

"""
    function generate(
    model::Transformer,
    tokenizer::Tokenizer,
    sampler::Sampler,
    prompt::String,
    extend::Bool=true,
    verbose::Bool=true,
    display_output::Bool=true,
    display_prompt::Bool=true,
)

Sample the next token (id) based on the logits and the sampler parameters.
"""
function generate(
    model::Transformer,
    tokenizer::Tokenizer,
    sampler::Sampler,
    prompt::String,
    extend::Bool=true,
    verbose::Bool=true,
    display_output::Bool=true,
    display_prompt::Bool=true,
)
    steps = model.config.seq_len

    prompt = replace(prompt, r"\s+" => " ")
    prompt = String(strip(prompt))

    # Encode the prompt
    prompt_tokens = encode(tokenizer, prompt)
    !isempty(prompt_tokens) || throw(error("something is wrong, expected at least 1 prompt token"))
    
    token = prompt_tokens[1]
    output = ""
    start = nothing
    pos = 1

    while pos <= steps
        logits = forward!(model, token, pos)
        if pos < length(prompt_tokens)
            token = prompt_tokens[pos + 1]
        else
            token = sampler(logits)
        end

        if token == 3 || token == 2 # EOS or BOS token.
            if display_output
                println()
            end
            break
        end

        decoded = decode(tokenizer, 1, token)
        output *= decoded

        pos += 1

        if !display_prompt
            if pos <= length(prompt_tokens)
                continue
            end
        end
        
        if display_output
            print(decoded)
        end

        # Initialize timer after the first iteration
        if start === nothing
            start = time_ns()
        end
    end

    # Continue generation if extend flag is set and sequence was cut off
    if extend && pos > steps
        prompt_new = output[div(length(output), 2):end]
        recursive_output = generate(
            model, tokenizer, sampler, prompt_new, true, false, display_output, false
        )
        recursive_output = replace(recursive_output, prompt_new => "")
        output *= recursive_output
    end

    # Report achieved tok/s
    if verbose
        if pos > 1 && start !== nothing
            elapsed_time = (time_ns() - start) / 1e9 # elapsed time in seconds
            println("\nachieved tok/s: ", pos / elapsed_time)
        end
    end

    return String(strip(output))
end