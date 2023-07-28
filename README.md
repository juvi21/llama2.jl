## llama2.jl

Tired of low level languages and ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure Julia? Great news, you can do it now at 'similiar' speeds in under 300 lines.

This is a fork of Andrej's [llama2.c](https://github.com/karpathy/llama2.c) and ported it to ( for now ) a little hacky version of julia.
This README is heavily inspired by the rust port [llama.rs](https://github.com/gaxler/llama2.rs)

## How to run?

1. Grab Andrej's baby Llama2 ([Orig instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset 

    ```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
    ```
2. Make sure you have the tokenizer binary - `tokenizer.bin` (if not see [tokenizer.py](tokenizer.py))
3. Run run.jl

    Single threaded:

    ```bash
    julia run.jl <model> <tokenizer> --temp [temperature]
    ```

   Multi-Threaded:In Progress
   CUDA: In Progress
   
## Contributions

Join the dark side and code in julia. 
Contributions are highly encouraged!

### Contriubtion IDEAS

- make it faster
- CUDA support
- Multi-Threaded support
