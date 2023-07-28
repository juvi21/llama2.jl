# llama2.jl
<p align="center">
  <img src="assets/jl_cute_lama.png" width="300" height="300" alt="Cute Llama">
</p>

Don't want to read? No problem got ya covered.
     ```bash
    git clone https://github.com/juvi21/llama2.jl && cd llama2.jl && wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin && julia jl_helpers/install_pkg.jl && julia run.jl stories15M.bin tokenizer.bin
      ```

Tired of low-level languages? Ever wanted to infer a baby [Llama 2](https://ai.meta.com/llama) model in pure Julia? Great news – you can now do so at good speeds in under 300 lines of Julia. 

This is a fork of Andrej's [llama2.c](https://github.com/karpathy/llama2.c) which has been ported to (for now) a slightly hacky version of Julia. This README is heavily inspired by the Rust port [llama.rs](https://github.com/gaxler/llama2.rs).

## How to run?

1. Grab Andrej's baby Llama2 (see the [original instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

    ```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
    ```
2. Ensure you have the tokenizer binary - `tokenizer.bin` (if not, see [tokenizer.py](tokenizer.py)).
3. Run `run.jl`:

    **Single-threaded:**

    ```bash
    julia run.jl <model> <tokenizer> --temp [temperature]
    ```

   **Multi-Threaded**: In Progress  
   **CUDA**: In Progress
   
## Contributions

Join the dark side and code in Julia. 
Contributions are highly encouraged!

**Contribution Ideas:**

- Make it faster.
- Add CUDA support.
- Introduce Multi-Threaded support.

## Art
All images in this README were created by Midjourney
