---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
- deepseek
- vllm
- speculative-decoding
- dspark
- sm_120
- cuda-kernels
- blackwell
- rtx-pro-6000
---

# DeepSeek-4.1-Flash-1017-BRRR

Recipe for [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on 4× RTX PRO 6000 with SSD ngram offload, heavily tuned for codegen decode speeds (especially at c1).

It adds fused decode kernels and a lot of model/hardware-specific optimisations on top of b12x from [Local Inference Lab](https://github.com/local-inference-lab); full props to them for the base image!

> **NOTE:**
> I did this for my personal use-case, so this is quite overfit to my hardware/requirements; e.g. no in-RAM ngram.
> I likely won't be maintaining unless there are significant issues or bugs.
> Feel free to use/adapt as you wish, Apache 2.0 licence

*[Follow my work on X!](https://x.com/fraserpricee)*

---

## Quickstart

0. Download the base `DeepSeek-V4.1-Flash-1017` weights (if you haven't already):

   ```bash
   hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir ~/models/DeepSeek-V4.1-Flash-1017
   ```

1. Download this recipe:

   ```bash
   hf download fraserprice/DeepSeek-4.1-Flash-1017-BRRR --local-dir ~/DeepSeek-4.1-Flash-1017-BRRR
   ```

2. Serve, pointing it at the weights:

   ```bash
   bash ~/DeepSeek-4.1-Flash-1017-BRRR/serve.sh ~/models/DeepSeek-V4.1-Flash-1017
   ```

It first checks GPU-to-GPU (P2P) transfers and, if your system isn't set up for them, shows what it needs to change and asks before changing anything (see [PCIe P2P](#pcie-p2p)). The first run builds the image, which pulls the base image and compiles the kernels (will take some time). Later runs reuse it so should be quicker. Add `--background` to keep it serving after you close the terminal; it then restarts on crash and after reboots.

This serves an OpenAI-compatible API (text, images and tool calls) as `DeepSeek-4.1-Flash-1017-BRRR` on port 8000. Check it with:

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"DeepSeek-4.1-Flash-1017-BRRR","messages":[{"role":"user","content":"Hello World!"}]}'
```

### Requirements

- 4× RTX PRO 6000 Blackwell (96 GB, any edition).
- PCIe 5.0 x16 P2P between all four GPUs (`serve.sh` checks this and offers to set it up).
- The weights on a local NVMe drive with ext4 or xfs (Engram tables are read from disk with `O_DIRECT`, so NFS and tmpfs won't work). Note that drive latency will impact decode speeds.
- x86_64 Linux, NVIDIA driver ≥ 615.65 (the image ships CUDA 13.4), Docker with the NVIDIA Container Toolkit.

#### PCIe P2P

`serve.sh` checks P2P between your GPUs on every run. If it isn't set up, it lists the system settings it needs to change and asks before changing them, then tells you if you need to reboot. If something is still wrong after that (e.g. your board only lets you turn the IOMMU off in the BIOS), it says what and asks whether to serve anyway.

Why: with the IOMMU on, P2P traffic gets routed through the CPU for address translation, and the NVIDIA driver disables or downgrades P2P on these cards (especially across root complexes) unless forced to. NCCL is already set up for this in the image (`NCCL_P2P_LEVEL=SYS`, `NCCL_P2P_READ_ENABLE=1`).

### Options

```
--gpus LIST      four GPU indices (default 0,1,2,3)
--port N         host port (default 8000)
--context N      max tokens per request (default 1048576)
--api-key KEY    require this key as a bearer token
--adaptive       enable DSpark adaptive verification (faster at high context and concurrency)
--background     keep serving in the background, restarting on crash and reboot
```

> NOTE: `--adaptive` turns on DSpark's adaptive verification, which uses its confidence head to verify fewer drafts per step when they're unlikely to be accepted. It's off by default: at low context and concurrency it's neutral to slightly slower, but at high context and concurrency it should improve decode speeds.

Anything after `--` goes to `vllm serve` and overrides the defaults in `entrypoint.sh`, e.g:

```bash
bash serve.sh ~/models/DeepSeek-V4.1-Flash-1017 --gpus 4,5,6,7 --port 9000 -- --gpu-memory-utilization 0.92
```

In the background, follow the logs with `docker logs -f deepseek-4.1-flash-1017-brrr` and stop it with `docker rm -f deepseek-4.1-flash-1017-brrr`.

## Results

Measured on my build: 4× RTX PRO 6000 Blackwell **Max-Q** (325 W), Threadripper 9960X, 64 GB RAM, Samsung 990 PRO. The 600 W editions should give higher numbers.
This was run with adaptive verification off.

> **NOTE:** My cooling setup is awful hence the short bench; if anyone wants to submit more/longer benches at different context or reasoning levels, feel free! See [Benchmarking](#benchmarking).

**Thinking off** (`uv run bench`; up to 2048 generated tokens, natural EOS)

```
     prompt       conc   TTFT p50      PP p1     PP p50     PP p99   PP total      TG p1     TG p50     TG p99   TG total  TG window     accept
      2,147          1     172 ms     12,648     12,452     12,309     12,473        738        703        681        706      1,403 5.94 (71%)
      2,147          4     489 ms     10,689      4,470      3,255     14,618        348        329        315      1,320      1,304 5.94 (71%)
      2,148          8     829 ms      8,748      2,589      1,842     15,052        242        231        218      1,847      1,279 5.97 (71%)
      2,147         16   1,333 ms      6,573      1,610      1,048     16,779        175        166        155      2,648      1,224 5.96 (71%)
     16,484          1     972 ms     16,998     16,953     16,844     16,943        713        684        679        691      1,687 5.99 (71%)
     16,484          4   2,362 ms     14,920      7,199      4,654     18,661        339        321        303      1,281      1,296 5.95 (71%)
     16,483          8   4,089 ms     14,266      4,072      2,369     18,984        240        226        216      1,806      1,188 5.99 (71%)
     16,484         16   7,624 ms     13,190      2,167      1,190     19,062        168        158        149      2,522      1,135 5.98 (71%)
     65,635          1   3,677 ms     17,916     17,852     17,771     17,856        680        675        651        669      1,553 5.97 (71%)
     65,635          4   9,165 ms     17,111      7,435      4,544     18,197        325        312        290      1,243      1,361 6.02 (72%)
     65,629          8  16,396 ms     16,842      4,051      2,261     18,140        230        210        199      1,686      1,224 5.95 (71%)
     65,635         16  30,893 ms     16,570      2,131      1,068     17,227        140        131        122      2,096      1,187 5.98 (71%)
```

**Thinking 25** (`uv run bench --thinking 25 -s 2048 16384 -r 3`)

```
     prompt       conc   TTFT p50      PP p1     PP p50     PP p99   PP total      TG p1     TG p50     TG p99   TG total  TG window     accept
      2,174          1     180 ms     12,189     12,093     11,515     11,922        541        507        492        507      6,291 4.27 (47%)
      2,174          4     488 ms     10,017      4,452      3,555     14,556        240        215        198        877      4,998 4.32 (47%)
      2,174          8     783 ms      8,662      2,787      1,905     15,753        170        142        129      1,159      4,396 4.32 (47%)
      2,174         16   1,358 ms      6,565      1,601      1,038     16,731        127         97         89      1,602      3,790 4.35 (48%)
     16,510          1     982 ms     16,815     16,809     16,795     16,806        528        498        496        507      5,089 4.31 (47%)
     16,509          4   2,371 ms     14,834      7,188      4,654     18,652        233        201        186        824      4,486 4.23 (46%)
     16,509          8   4,101 ms     14,190      4,067      2,369     18,964        179        136        125      1,115      4,480 4.28 (47%)
     16,510         16   7,564 ms     13,163      2,190      1,194     19,112        106         90         81      1,462      3,850 4.32 (47%)
```

> Anecdotally, I've found much better quality per unit of wall time at 25% thinking than at the standard `low` (50%). Give it a try!

### Accuracy

I ran a quick KLD check against stock b12x vLLM (same image, flags and DSpark config); it matches stock's KLD against itself within noise.

| Concurrency | Stock vs Stock | BRRR vs BRRR | BRRR vs Stock |
| ----------: | -------------: | -----------: | ------------: |
|           1 |         0.0099 |       0.0117 |        0.0107 |
|           2 |         0.0133 |       0.0102 |        0.0123 |
|           4 |         0.0111 |       0.0094 |        0.0111 |
|           8 |         0.0120 |       0.0097 |        0.0108 |
|          16 |         0.0116 |       0.0098 |        0.0113 |
|     **All** |     **0.0116** |   **0.0102** |    **0.0113** |

Full-vocab KLD over the same 1,024 forced tokens, two runs per concurrency level.
BRRR vs Stock is within noise of the self-comparisons (+0.0004 ± 0.0004), and every column agrees on the top-1 token 96% of the time.

## Benchmarking

```bash
bash serve.sh ~/models/DeepSeek-V4.1-Flash-1017
uv run bench                # reasoning off
uv run bench --thinking 25  # reasoning 25%
```

It's a quick and dirty script to measure codegen speed. It pads a coding prompt to 2K, 16K and 64K tokens (with a nonce to bust the prefix cache) and runs 5 rounds of each at concurrency 1, 4, 8 and 16. Before each round it waits for the hottest GPU to cool to 70 °C, because my cooling is god-awful lmao (change it with `--cool-to`).

vLLM interleaves chunked prefill with decode, so each round sends its prompts twice: first with `max_tokens: 1` to time pure prefill (TTFT, PP), then again from cache so every stream starts decoding together. TG only counts while all streams are decoding, and `TG window` is the median tokens per stream in that window. Output runs to natural EOS, so speculative acceptance should reflect real code generation.

## How it works

It's built on the `ghcr.io/local-inference-lab/vllm` image, which provides the b12x attention, MoE and linear backends and DSpark speculative decoding. Everything added on top targets decode for this one model at TP=4, with its shapes compiled into the kernels. Prefill and anything outside a kernel's limits fall back to stock b12x (there's probably room for improvement here!).

- **No vLLM fork.** `brrr.pth` installs an import hook that patches vLLM classes in place as they load (`patches.py`), and `vllm.patch` edits four vLLM files at build time where a change has to land inside a function body. The kernels in `src/brrr/kernels/` are compiled for `sm_120a` during `docker build`.
- **Fused decoder layer** (`decoder.cu`). For decode batches of up to 128 tokens, everything after attention runs as one persistent kernel: output projection, hyper-connections, router, shared and routed experts, and both TP all-reduces. The all-reduces run chunk by chunk over CUDA IPC, overlapping the matmuls still in flight. At 8 tokens or fewer, a second kernel also fuses everything before attention.
- **Attention.** `sparse_decode.cu` gives each query row an 8-CTA cluster that reads FP8 keys straight from the paged cache. `candidates.cu` replaces the indexer's top-k with a radix select, and `skinny.cu` is an FP8 GEMM for 8 tokens or fewer.
- **Communication and sampling.** `phased_all_reduce.cu` handles the 16 KB–1 MB all-reduces that b12x's custom all-reduce turns down. `vp_sampling.py` keeps DSpark drafting and rejection sampling split by vocab across GPUs, so only a few words per row cross GPUs instead of full logits. It handles any temperature, but requests using top-p, top-k, min-p, penalties, logit bias, logprobs or structured output fall back to stock sampling.
- **Prompt fixes.** `vllm.patch` also backports DeepSeek's reference-encoder rule that a message holding only tool results doesn't start a new turn (so reasoning inside a tool loop isn't dropped), and sorts parallel tool results by call order before multimodal parsing so images stay with the right result.
- **Engram on NVMe** (`engram_nvme.py`, `engram_nvme.cpp`). The Engram tables for layers 1 and 14 are about 98 GB each and stay on disk. Each layer's rows are requested before it runs and read through io_uring while the layers in front of it compute.

The serving defaults are in `entrypoint.sh`: TP4, FP8 KV cache, DSpark with 7 draft tokens, 1M context, 16 sequences, a 4096-token batch budget per step, and CUDA graphs up to batch 128.

## License

[Apache-2.0](LICENSE). Use, modify and ship it however you like, commercially or not; just keep the [NOTICE](NOTICE) file with any copies or derivatives so the credit stays attached.

This builds heavily on [vLLM](https://github.com/vllm-project/vllm), [Local Inference Lab's vLLM fork](https://github.com/local-inference-lab/vllm), [b12x](https://github.com/local-inference-lab/b12x) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer), all Apache-2.0 and the copyright of their respective authors.
