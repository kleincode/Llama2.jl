"""
$(TYPEDSIGNATURES)

Generate a sequence based on a given language model, tokenizer, sampler and prompt.

There are several optional boolean flags:
* `verbose::Bool`: Print the achieved tokens/s
* `display_output::Bool`: Print the output
* `display_prompt::Bool`: Print the prompt. Ignored if `display_output` is `false`.
* `max_steps::Int`: Maximum number of generation steps.

llama2.c correspondence: generation loop (l. 729-783)
"""
function generate(
    model::Transformer{T},
    tokenizer::Tokenizer,
    sampler::Sampler{T},
    prompt::String;
    verbose::Bool=true,
    display_output::Bool=true,
    display_prompt::Bool=true,
    max_steps::Int=-1,
) where {T<:Real}
    prompt = replace(prompt, r"\s+" => " ")
    prompt = String(strip(prompt))

    # Encode the prompt
    prompt_tokens = encode(tokenizer, prompt)
    !isempty(prompt_tokens) ||
        throw(error("something is wrong, expected at least 1 prompt token"))

    len_prompt = length(prompt_tokens)
    max_steps < 0 && (max_steps = model.config.seq_len - len_prompt) # the max number of tokens to generate (excluding prompt)
    steps = min(max_steps + len_prompt, model.config.seq_len) # max number of tokens to generate in this call

    token = prompt_tokens[1]
    output = ""
    start = nothing
    pos = 1

    while pos <= steps
        logits = forward!(model, token, pos)
        if pos < len_prompt
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
            if pos <= len_prompt
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

    steps_left = max_steps + min(len_prompt, steps) - pos

    # Continue generation if sequence was cut off and there are steps left
    if pos > steps && steps_left > 0
        prompt_new = output[div(length(output), 2):end]
        recursive_output = generate(
            model,
            tokenizer,
            sampler,
            prompt_new;
            verbose=false,
            display_output=display_output,
            display_prompt=false,
            max_steps=steps_left,
        )
        recursive_output = replace(recursive_output, prompt_new => "")
        output *= recursive_output
    end

    # Report achieved tok/s
    if verbose && pos > 1 && start !== nothing
        elapsed_time = (time_ns() - start) / 1e9 # elapsed time in seconds
        println("\nachieved tok/s: ", pos / elapsed_time)
    end

    return String(strip(output))
end