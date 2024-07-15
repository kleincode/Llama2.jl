"""
$(TYPEDSIGNATURES)

Used to chat.

In llama2.c this is not safely implemented and more a proof of concept

llama2.c correspondence Config (l.802-884)
"""
function chat(
    transformer::Transformer{T},
    tokenizer::Tokenizer{T},
    sampler::Sampler{T};
    cli_user_prompt::String="",
    cli_system_prompt::String="",
    steps::Integer=256,
) where {T<:Real}
    system_prompt::String = ""
    user_prompt::String = ""
    rendered_prompt::String = ""
    prompt_tokens::Vector{Int} = []
    num_prompt_tokens::Integer = 0
    user_idx::Integer = 1

    # start the main loop 
    user_turn::Bool = true   # stores next token in the sequence
    next::Integer = 0        # stores current token
    token::Integer = 0
    pos::Integer = 1     # position in the sequence

    while pos <= steps
        # user's turn to contribute token to the dialog
        if user_turn == true
            """
            if pos == 1
                if cli_system_prompt === ""
                    print("Enter system prompt (optional):")
                    system_prompt = readline()
                else
                    system_prompt = cli_system_prompt
                end
            end"""

            if pos == 1 && cli_user_prompt !== ""
                user_prompt = cli_user_prompt
            else
                println("User: ")
                # wanted to use readline but that did not quite work
                user_prompt = parse(UInt8, readline())
            end
            # render user/system prompts into the Llama 2 Chat schema
            rendered_prompt = "[INST]" * user_prompt * "[/INST]"
            """
            if pos == 1 && system_prompt !== ""
                rendered_prompt =
                    "[INST] <<SYS>>\n" *
                    system_prompt *
                    "\n<</SYS>>\n\n" *
                    user_prompt *
                    "[/INST]"
            else
                rendered_prompt = "[INST]" * user_prompt * "[/INST]"
            end
            """
            # encode the rendered prompt into tokens
            prompt_tokens = encode(tokenizer, rendered_prompt)
            num_prompt_tokens = length(prompt_tokens)
            user_idx = 1
            user_turn = false
            print("Assistant: ")
        end

        # determine the token to pass into the transformer next
        if user_idx < num_prompt_tokens
            token = prompt_tokens[user_idx]
            user_idx += 1
        else
            token = next
        end
        # EOS token (=3) ends the Assistant turn
        if token == 3
            user_turn = true
        end

        # forward the transformer to get logits for the next token
        logits = forward!(transformer, token, pos)
        next = sampler(logits)
        pos += 1

        if user_idx >= num_prompt_tokens && next != 3
            # assistant is responding, print its output
            piece = decode(tokenizer, token, next)
            print(piece)
        end
        if next == 3
            print("\n")
        end
    end
    return print("\n")
end