# llama2.jl
<p align="center">
  <img src="assets/jl_cute_lama.png" width="300" height="300" alt="Cute Llama">
</p>

Tired of low-level languages? Ever wanted to infer a baby [Llama 2](https://ai.meta.com/llama) model in pure Julia? Great news – you can now do so at in under 300 lines of Julia. 

This is a fork of Andrej's [llama2.c](https://github.com/karpathy/llama2.c) which has been ported to (for now) a slightly hacky version of Julia. This README is heavily inspired by the Rust port [llama.rs](https://github.com/gaxler/llama2.rs).

**Don't want to read? Got ya back!**     

```bash
git clone https://github.com/juvi21/llama2.jl && cd llama2.jl && wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin && julia jl_helpers/install_pkg.jl && julia run.jl stories15M.bin tokenizer.bin
```

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

## Performance
On my current workstation, the performance is quite fast. However, I have been away visiting my parents for a few days, so I only had the opportunity to test it on one of my very first and less powerful station. More testing is coming soon!
**NOTE**: I compiled llama2.c with the provided command in Andrej's README which is only the basic one to get started and not very optimized.

    
    gcc -O3 -o run run.c -lm
    
    
| system                   | model          | llama2.c            | llama2.jl            |
| ------------------------ | -------------- | ------------------ | ------------------- |
| Ubuntu 22.04 AMD Ryzen 2600 | stories15M.bin | 85.418752 tok/s   | 257.445516 tok/s    |
| Ubuntu 22.04 AMD Ryzen 2600 | stories42M.bin | 30.761836 tok/s   | 92.567484 tok/s     |
| Ubuntu 22.04 AMD Ryzen 2600 | stories110.bin | 11.585283 tok/s   | 38.543434 tok/s     |

## Contributions

Join the dark side and code in Julia. 
Contributions are highly encouraged!

**Contribution Ideas:**

- Make it faster.
- Add CUDA support.
- Introduce Multi-Threaded support.

## Art
All images in this README were created by Midjourney
