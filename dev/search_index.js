var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Llama2","category":"page"},{"location":"#Llama2","page":"Home","title":"Llama2","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Llama2.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Llama2]","category":"page"},{"location":"#Llama2.Config","page":"Home","title":"Llama2.Config","text":"Configuration of initial parameters\n\nllama2.c correspondence Config (l.19)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.RunState","page":"Home","title":"Llama2.RunState","text":"Initial parameters for the run state\n\nllama2.c correspondence: RunState (l. 50)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.RunState-Tuple{Config}","page":"Home","title":"Llama2.RunState","text":"Initialization of RunState based on Config\n\ninitialized as undefined arrays\n\nllama2.c correspondence: mallocrunstate (l. 77)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Tokenizer","page":"Home","title":"Llama2.Tokenizer","text":"Used for mapping from strings to token arrays (Int vectors) and back.\n\nllama.c correspondence: Tokenizer (l. 372)\n\nindextotoken = vocab\ntokentoindex = sorted_vocab\nremoved maxtokenlength (not required in Julia)\nremoved byte_pieces (not required in Julia)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.Tokenizer-Tuple{String, Int64}","page":"Home","title":"Llama2.Tokenizer","text":"Tokenizer(tokenizer_path::String, vocab_size::Int)\n\nConstructs a Tokenizer by loading the vocabulary from a file in the llama2.c format. The vocabulary size must be known from the config.\n\nllama.c correspondence: build_tokenizer (l. 385)\n\n\n\n\n\n","category":"method"},{"location":"#Llama2.Transformer","page":"Home","title":"Llama2.Transformer","text":"Initial parameters for the transformer\n\nllama2.c correspondence: Transformer (l. 67)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.TransformerWeights","page":"Home","title":"Llama2.TransformerWeights","text":"Initial parameters for the transformer weights\n\nllama2.c correspondence: TransformerWeights (l. 29)\n\n\n\n\n\n","category":"type"},{"location":"#Llama2.TransformerWeights-Tuple{Config}","page":"Home","title":"Llama2.TransformerWeights","text":"Initialize transformer weights based on Config\n\ninitialized as undefined arrays\n\nllama2.c correspondence: memorymapweights (l. 111)\n\n\n\n\n\n","category":"method"}]
}
