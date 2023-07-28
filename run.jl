# This is a port of Andrej Karpathy's https://github.com/karpathy/llama2.c to Julia. I'm quite new to Julia so contributions are highly encouraged

import Base.@kwdef
using LinearAlgebra
using Serialization
using StatsBase
using Printf
using ArgParse

@kwdef struct Config
    dim::Int
    hidden_dim::Int
    n_layers::Int
    n_heads::Int
    n_kv_heads::Int
    vocab_size::Int
    seq_len::Int
end

@kwdef struct TransformerWeights
    token_embedding_table::Matrix{Float32}
    rms_att_weight::Matrix{Float32}
    rms_ffn_weight::Matrix{Float32}
    wq::Array{Float32,3}
    wk::Array{Float32,3}
    wv::Array{Float32,3}
    wo::Array{Float32,3}
    w1::Array{Float32,3}
    w2::Array{Float32,3}
    w3::Array{Float32,3}
    rms_final_weight::Vector{Float32}
    freq_cis_real::Matrix{Float32}
    freq_cis_imag::Matrix{Float32}
end

TransformerWeights(config::Config) = TransformerWeights(;
    token_embedding_table = zeros(Float32, config.dim, config.vocab_size),
    rms_att_weight        = zeros(Float32, config.dim, config.n_layers),
    rms_ffn_weight        = zeros(Float32, config.dim, config.n_layers),
    wq                    = zeros(Float32, config.dim, config.dim, config.n_layers),
    wk                    = zeros(Float32, config.dim, config.dim, config.n_layers),
    wv                    = zeros(Float32, config.dim, config.dim, config.n_layers),
    wo                    = zeros(Float32, config.dim, config.dim, config.n_layers),
    w1                    = zeros(Float32, config.dim, config.hidden_dim, config.n_layers),
    w2                    = zeros(Float32, config.hidden_dim, config.dim, config.n_layers),
    w3                    = zeros(Float32, config.dim, config.hidden_dim, config.n_layers),
    rms_final_weight      = zeros(Float32, config.dim),
    freq_cis_real         = zeros(Float32, (config.dim ÷ config.n_heads) ÷ 2, config.seq_len),
    freq_cis_imag         = zeros(Float32, (config.dim ÷ config.n_heads) ÷ 2, config.seq_len)
)

@kwdef struct RunState
    x::Vector{Float32}
    xb::Vector{Float32}
    xb2::Vector{Float32}
    hb::Vector{Float32}
    hb2::Vector{Float32}
    q::Vector{Float32}
    k::Vector{Float32}
    v::Vector{Float32}
    att::Vector{Float32}
    logits::Vector{Float32}
    key_cache::Array{Float32,3}
    value_cache::Array{Float32,3}
end

RunState(config::Config) = RunState(;
    x           = zeros(Float32, config.dim),
    xb          = zeros(Float32, config.dim),
    xb2         = zeros(Float32, config.dim),
    hb          = zeros(Float32, config.hidden_dim),
    hb2         = zeros(Float32, config.hidden_dim),
    q           = zeros(Float32, config.dim),
    k           = zeros(Float32, config.dim),
    v           = zeros(Float32, config.dim),
    att         = zeros(Float32, config.seq_len),
    logits      = zeros(Float32, config.vocab_size),
    key_cache   = zeros(Float32, config.dim, config.seq_len, config.n_layers),
    value_cache = zeros(Float32, config.dim, config.seq_len, config.n_layers),
)

function checkpoint_init_weights!(w::TransformerWeights, f::IOStream)
    fields = [w.token_embedding_table, w.rms_att_weight, w.wq, w.wk, w.wv, w.wo, w.rms_ffn_weight, w.w1, w.w2, w.w3, w.rms_final_weight, w.freq_cis_real, w.freq_cis_imag]
    for field in fields
        read!(f, field)
    end
    return nothing
end

read_config(f::IOStream) = Config((read(f, Int32) for _ in 1:7)...)


function rmsnorm!(o, x, weight)
    len_x_inv = 1.0 / length(x)
    ss = dot(x, x) * len_x_inv
    ss += 1f-5
    inv_ss = inv(sqrt(ss))
    o .= weight .* (x .* inv_ss)
    return nothing
end

function softmax!(x)
    max_val = maximum(x)
    x .= exp.(x .- max_val)
    norm_factor = inv(sum(x))
    x .*= norm_factor
    return nothing
end


function fast_copyto!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    size(dest) == size(src) || throw(DimensionMismatch("Source and destination arrays have different sizes"))
    if ndims(dest) == 1 && ndims(src) == 1
        @inbounds @simd for i in eachindex(dest, src)
            dest[i] = src[i]
        end
    else
        @inbounds for i in eachindex(dest, src)
            dest[i] = src[i]
        end
    end
    return dest
end

@views function attention!(q, k, v, pos, config, s, w, l)
    dim = config.dim
    head_size = dim ÷ config.n_heads
    sqrt_head_size = sqrt(Float32(head_size))

    for h in 1:config.n_heads
        start_idx = (h-1) * head_size + 1
        end_idx = h * head_size
        q_h = q[start_idx:end_idx]

        @inbounds @simd for t in 1:pos
            k_h = s.key_cache[start_idx:end_idx, t, l]
            s.att[t] = dot(q_h, k_h) / sqrt_head_size
        end

        softmax!(s.att[1:pos])

        mul!(
            s.xb[start_idx:end_idx],
            s.value_cache[start_idx:end_idx, 1:pos, l],
            s.att[1:pos]
        )
    end
end

@views function feed_forward!(x, s, w, l, hidden_dim)
    rmsnorm!(s.xb, x, w.rms_ffn_weight[:, l])
    mul!(s.hb, w.w1[:, :, l]', s.xb)
    mul!(s.hb2, w.w3[:, :, l]', s.xb)
    @inbounds @simd for i in 1:hidden_dim
        s.hb[i] *= 1f0 / (1f0 + exp(-s.hb[i]))
    end
    s.hb .*= s.hb2
    mul!(s.xb, w.w2[:, :, l]', s.hb)
    x .+= s.xb
end

@views function freq_transform!(q, k, head_size, freq_cis_real_row, freq_cis_imag_row)
    @inbounds @simd for h in 1:head_size ÷ 2
        idx1, idx2 = 2*h-1, 2*h
        q0, q1 = q[idx1], q[idx2]
        k0, k1 = k[idx1], k[idx2]
        fcr, fci = freq_cis_real_row[h], freq_cis_imag_row[h]
        
        tmp1, tmp2 = q0 * fcr, q1 * fci
        q[idx1] = tmp1 - tmp2
        q[idx2] = q0 * fci + q1 * fcr
        
        tmp1, tmp2 = k0 * fcr, k1 * fci
        k[idx1] = tmp1 - tmp2
        k[idx2] = k0 * fci + k1 * fcr
    end
end

@views function transformer!(token::Int, pos::Int, config::Config, s::RunState, w::TransformerWeights)
    x = s.x
    dim = config.dim
    hidden_dim = config.hidden_dim
    head_size = dim ÷ config.n_heads

    fast_copyto!(x, w.token_embedding_table[:, token])

    freq_cis_real_row = w.freq_cis_real[:, pos]
    freq_cis_imag_row = w.freq_cis_imag[:, pos]

    for l in 1:config.n_layers
        rmsnorm!(s.xb, x, w.rms_att_weight[:, l])
        mul!(s.q, w.wq[:, :, l]', s.xb)
        mul!(s.k, w.wk[:, :, l]', s.xb)
        mul!(s.v, w.wv[:, :, l]', s.xb)

        for h in 1:config.n_heads
            start_idx = (h-1) * head_size + 1
            q = s.q[start_idx:(start_idx + head_size - 1)]
            k = s.k[start_idx:(start_idx + head_size - 1)]
            freq_transform!(q, k, head_size, freq_cis_real_row, freq_cis_imag_row)
        end

        fast_copyto!(s.key_cache[:, pos, l], s.k)
        fast_copyto!(s.value_cache[:, pos, l], s.v)

        attention!(s.q, s.k, s.v, pos, config, s, w, l)

        mul!(s.xb2, w.wo[:, :, l]', s.xb)
        x .+= s.xb2

        feed_forward!(x, s, w, l, hidden_dim)
    end

    rmsnorm!(x, x, w.rms_final_weight)
    mul!(s.logits, w.token_embedding_table', x)

    return nothing
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "checkpoint"
        help = "filename of the model checkpoint"
        arg_type = AbstractString
        required = true

        "tokenizer"
        help = "filename of the tokenizer"
        arg_type = AbstractString
        required = true

        "--temp"
        help = "temperature setting for the model"
        arg_type = Float32
        default = 0.9f0
    end

    parsed_args = parse_args(ARGS, s)

    checkpoint_filename = parsed_args["checkpoint"]
    tokenizer_filename = parsed_args["tokenizer"]
    temperature = parsed_args["temp"]

    config = nothing
    weights = nothing

    open(checkpoint_filename) do file
        config = read_config(file)
        weights = TransformerWeights(config)
        checkpoint_init_weights!(weights, file)
    end

    vocab = open(tokenizer_filename) do file
        [begin
            len = read(file, Int32)
            read(file, len)
        end for _ in 1:config.vocab_size]
    end

    state = RunState(config)
    time_start = time_ns()
    token = 1

    for pos in 1:config.seq_len
        transformer!(token, pos, config, state, weights)

        if temperature == 0f0
            next = argmax(state.logits)
        else
            state.logits ./= temperature
            softmax!(state.logits)
            next = wsample(1:config.vocab_size, state.logits)
        end

        print(String(copy(vocab[next])))
        token = next
    end
    println()

    time_end = time_ns()
    @printf "tok/s: %f\n" config.seq_len / (time_end - time_start)*1e9

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end