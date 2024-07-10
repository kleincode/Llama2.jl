"""
$(TYPEDSIGNATURES)

Used to chat.

In llama2.c this is not safely implemented and more a proof of concept

llama2.c correspondence Config (l.802-884)
"""
function chat(transformer::Transformer, tokenizer::Tokenizer, sampler::Sampler)
    system_prompt::String = ""
    user_prompt::String = ""
    rendered_prompt::String = ""
    num_prompt_tokens::Int32 = 0
    user_idx::Int32 = 1

    # start the main loop 
    user_turn::Bool = true   # stores next token in the sequence
    next::Int32 = 0        # stores current token
    token::Int32 = 0
    pos::Int32 = 0     # position in the sequence

    while pos < steps
        # user's turn to contribute token to the dialog
        if user_turn
            if pos == 0
                if cli_system_prompt === nothing
                    print("Enter system prompt (optional):")
                    system_prompt = readline()
                else
                    system_prompt = cli_system_prompt
                end
            end

            if pos == 0 && cli_user_prompt !== nothing
                user_prompt = cli_user_prompt
            else
                println("User: ")
                user_prompt = readline()
            end
            # render user/system prompts into the Llama 2 Chat schema
            if pos == 0 && system_prompt !== nothing
                rendered_prompt = "[INST] <<SYS>>\n" * system_prompt * "\n<</SYS>>\n\n" * user_prompt * "[/INST]"
            else
                rendered_prompt = "[INST]" * user_prompt * "[/INST]"
            end
            # encode the rendered prompt into tokens
            prompt_tokens = encode(tokenizer, rendered_prompt)
            num_prompt_tokens = size(prompt_tokens)
            user_idx = 0
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
        if token == 3 user_turn = true end 

        # forward the transformer to get logits for the next token
        logits= forward!(transformer, token, pos)
        next = sampler(logits)
        pos += 1

        if user_idx >= num_prompt_tokens && next != 3
            # assistant is responding, print its output
            piece = decode(tokenizer, token, next)
            print(piece)
        end
        if next == 3 print("\n") end
    end
    print("\n")
end