using Downloads

"""
    get_stories15M()::String
Downloads stories15M.bin if not already downloaded and returns the path on the local disk.
"""
function get_stories15M()
    llama_file = "../bin/transformer/stories15M.bin"
    if !isfile(llama_file)
        println("Downloading stories15M.bin...")
        Downloads.download(
            "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
            llama_file,
        )
        println("Download complete!")
    end
    return llama_file
end