using Llama2
using Test

# Define the tests
@testset "generate" begin

    # Initialize model and tokenizer
    config, weights = read_karpathy(get_stories15M())
    state = RunState{Float32}(config)
    transformer = Transformer{Float32}(config, weights, state)
    tokenizer = Tokenizer("../bin/tokenizer/tokenizer.bin", 32000)
    sampler_1 = Sampler{Float32}(0.5f0, 1.0f0, 420)
    sampler_2 = Sampler{Float32}(1.0f0, 0.2f0, 187)

    @testset "Empty prompt" begin
        output = generate(transformer, tokenizer, sampler_2, "", verbose=false, display_output=false, display_prompt=false)
        @test typeof(output) == String
        @test !isempty(output)
    end

    @testset "Single word prompt" begin
        output = generate(transformer, tokenizer, sampler_1, "Hello", verbose=false, display_output=false, display_prompt=false)
        @test typeof(output) == String
        @test !isempty(output)
    end

    @testset "Long prompt" begin
        long_prompt = "Once upon a time, in a faraway land, there was a small village surrounded by lush green forests and flowing rivers. The villagers were known for their"
        output = generate(
            transformer, tokenizer, sampler_1, long_prompt, verbose=false, display_output=false, display_prompt=false
        )
        @test typeof(output) == String
        @test !isempty(output)
    end

    @testset "Special characters prompt" begin
        special_char_prompt = "#%^&*()_+123-=[]{}|?;':,./<>"
        output = generate(
            transformer, tokenizer, sampler_1, special_char_prompt, verbose=false, display_output=false, display_prompt=false
        )
        @test typeof(output) == String
        @test !isempty(output)
    end

    @testset "display output & verbose" begin
        display_prompt = "Once upon a time, there was a little language model named Llama."
        output = generate(transformer, tokenizer, sampler_2, display_prompt)
        @test typeof(output) == String
        @test !isempty(output)
    end

    @testset "cutoff at max_steps" begin
        output = generate(
            transformer,
            tokenizer,
            sampler_1,
            "",
            verbose=false, 
            display_output=false, 
            display_prompt=false,
            max_steps=10,
        )
        @test output == "Once upon a time, there was a little girl named"
    end

    @testset "extend" begin
        sampler_1 = Sampler{Float32}(0.5f0, 1.0f0, 420)
        output = generate(
            transformer,
            tokenizer,
            sampler_1,
            "The quick brown fox jumps over",
            verbose=false, 
            display_output=false, 
            display_prompt=false
        )
        @test endswith(output, "They play")

        sampler_1 = Sampler{Float32}(0.5f0, 1.0f0, 420)
        output_extend = generate(
            transformer,
            tokenizer,
            sampler_1,
            "The quick brown fox jumps over",
            verbose=false, 
            display_output=false, 
            display_prompt=false,
            max_steps=1000,
        )
        @test endswith(output_extend, "They play together and have fun.")

        @testset "Extend Very long prompt" begin
            very_long_prompt = "Once upon a time, in a faraway land, there was a small village surrounded by lush green forests and flowing rivers. 
            The villagers were known for their kindness and generosity. They lived in harmony with nature and all the creatures that inhabited the land. The village was ruled by a wise and just king who was loved by all his subjects. 
            One day, a terrible dragon appeared and began to terrorize the village. The king called upon the bravest knights in the kingdom to slay the dragon and save the village. 
            The knights set out on a perilous journey to find the dragon's lair and put an end to its reign of terror. After many days of searching, they finally found the dragon's lair deep in the heart of the forest. 
            The knights bravely fought the dragon and after a fierce battle, they emerged victorious. The dragon was slain and the village was saved. 
            The king rewarded the knights with great riches and honor, and they were hailed as heroes throughout the land. The villagers celebrated their victory with a grand feast and a day of festivities. 
            From that day on, the village was at peace and the memory of the dragon faded into legend. But"

            output = generate(
                transformer, tokenizer, sampler_2, very_long_prompt, verbose=false, display_output=false, display_prompt=false
            )
            @test typeof(output) == String
            @test !isempty(output)

            output_extend = generate(
                transformer,
                tokenizer,
                sampler_2,
                very_long_prompt,
                verbose=false, 
                display_output=false, 
                display_prompt=false,
                max_steps=1000,
            )
            @test typeof(output_extend) == String
            @test !isempty(output_extend)
            @test length(output_extend) > length(output)
        end
    end
end