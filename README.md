# MOSS-Image on Hygon DCU (gfx936)

**First public run report, measured benchmarks, and a deployment harness for
running MOSS-Image on Hygon BW / DCU hardware.**

[中文说明](README.zh-CN.md)

---

## What this is, and what it is not

**Not a port.** MOSS-Image runs on Hygon DCU with *zero* changes to model code:
no patched kernels, no workarounds, no vendored forks. The upstream
`pipeline_moss_image.py` and the upstream `transformers` Qwen3.5 implementation
work as shipped.

What was actually contributed:

1. **First-run confirmation.** As of this writing there is no public report of
   MOSS-Image running on ROCm, AMD, or Hygon hardware. Searches of the upstream
   issue tracker and Hugging Face discussions for `rocm`, `amd`, `gcu`, `dcu`,
   and `hygon` return nothing. This repository documents a working configuration.
2. **Two hidden landmines checked and cleared.** See
   [Numerical hazards](#numerical-hazards). Both are failure modes that produce
   *wrong output without errors*, which is the worst kind.
3. **A real bug found in the serving layer.** Multi-threaded access to the HIP
   runtime from HTTP handler threads deadlocks. See
   [The deadlock](#the-deadlock-on-rocmpdtk). This affects anyone serving any
   diffusion model from a web UI on these drivers.
4. **Benchmarks and tuned settings** on real hardware, so the next person does
   not have to re-derive them.

If you are looking for "how to make MOSS-Image work on DCU", the answer is:
install the vendor PyTorch, install `diffusers` and `transformers`, run it. The
value here is knowing that answer up front, plus the traps to avoid.

---

## Hardware and software measured

| Item | Value |
|---|---|
| Accelerator | 1× Hygon HCU ("BW"), 80 CUs, gfx936 |
| VRAM | 65520 MiB (64.0 GiB) |
| Driver | 6.3.31-V1.5.3.beta |
| Toolchain | DTK 26.04 |
| PyTorch | `2.7.1+das.opt1.dtk2604` (vendor build, HIP 6.3.26093) |
| diffusers | 0.43.0 |
| transformers | 5.23.0 |
| Host limits | 59 GiB RAM, 15 CPU cores (cgroup), 1.8 TB disk |

### Measured compute (not spec sheet)

| Test | Result | A100 80GB reference |
|---|---|---|
| bf16 matmul 8192³ | **307.95 TFLOP/s** | 312 TFLOP/s dense |
| fp16 matmul 8192³ | 307.70 TFLOP/s | 312 TFLOP/s |
| fp32 matmul 4096³ | 54.48 TFLOP/s | 19.5 (no TF32) |
| Memory copy bandwidth | **1353.6 GB/s** | 2039 GB/s |
| bf16 SDPA, seq 4096 | 172.33 TFLOP/s | — |

bf16 throughput is A100-class; bandwidth is about two thirds. Diffusion is
compute-heavy and relatively bandwidth-insensitive, which is why this card does
well here.

Reproduce with `tools/bench_bw.py`.

---

## Numerical hazards

Both issues below silently produce wrong images rather than raising. Both were
tested here; see `tools/moss_te_check.py` and `tools/moss_timestep_check.py`.

### 1. Qwen3.5 GDN linear attention without optimized kernels

The text encoder is Qwen3.5-9B, whose stack is three Gated DeltaNet (linear
attention) layers per full-attention layer. The fast path needs `fla`
(flash-linear-attention) and `causal_conv1d`.

Neither is a hard dependency. `transformers` uses
`use_kernel_func_from_hub_with_fallback`, so when the packages are absent it
falls back to a reference PyTorch implementation and logs:

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch
implementation because `flash-linear-attention` is not installed.
This is correct but much slower; install `flash-linear-attention` for the
optimized kernel.
```

Measured on this card with the pure-PyTorch path:

- no NaN, no Inf, no zero-collapse
- output std ≈ 2.0, per-position std 0.52–29.1
- cross-prompt cosine similarity 0.73 / 0.87 / 0.75 (no embedding collapse)
- first text encode **6.64 s**, second call **0.10 s**

The cost is latency, not correctness. This matters because an earlier project on
the same card hit a `fla` build whose kernels returned zeros in the forward pass
and NaN in backward — so "it imports" is not proof. The probes in `tools/`
are what establish correctness.

### 2. float64 timestep embedding

The upstream pipeline computes the timestep sinusoid in float64:

```python
freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
phase = times.double()[:, None] * freq
```

On backends that do not support float64 this can **silently return zeros**,
making every timestep identical and the output pure noise, with no error raised.
It is a documented failure on MPS and on some NPU/XPU stacks.

On this Hygon card float64 works correctly:

```
input times: [1000.0, 976.0, 500.0, 24.0, 0.0]
per-timestep L2 norm: [11.314, 11.314, 11.314, 11.314, 11.314]
```

If you target a different accelerator, run `tools/moss_timestep_check.py` before
trusting any output. A zero norm means every image you generate is noise.

---

## Performance

MOSS-Image, 1024×1024, `uncond_skip_layers=[22]`, model resident in VRAM.

| Steps | Guidance | Time | Per step | Peak VRAM |
|---|---|---|---|---|
| 20 | 3.0 | **19.8 s** | 0.66 s | 28.2 GiB |
| 50 | 3.0 | 45.9 s | 0.79 s | 28.2 GiB |
| 20 | 1.0 | **9.6 s** | 0.48 s | 28.2 GiB |

Other resolutions, 20 steps at guidance 3.0:

| Resolution | Time | Peak VRAM |
|---|---|---|
| 1328×1328 | 37.6 s | 31.3 GiB |
| 1536×1024 | 33.3 s | 30.2 GiB |

Model residency is 25.4 GiB; loading takes 8–20 s depending on page cache.

### Why it is slower than a single 10 GB transformer

"10 GB" refers only to the DiT. The full pipeline is 26 GB:

| Component | On disk | Notes |
|---|---|---|
| transformer (DiT, 42 layers, dim 3072) | 9.5 GB | the 10 GB figure |
| text_encoder (Qwen3.5-9B, 8.39B params) | 16 GB | as large as the DiT |
| vae (FLUX.2 small) | 238 MB | fp32 by design |

Two independent reasons for the latency:

1. **CFG doubles the work.** With `guidance_scale != 1`, every step runs two
   DiT forwards (conditional and unconditional). Setting guidance to 1.0 halves
   the time — from 19.8 s to 9.6 s at 20 steps.
2. **The first text encode costs 6.6 s** because the GDN fallback is slow.

For reference, Qwen-Image 2.1 at 1024² does 0.40 s/step at `cfg=1.0` because it
runs one forward per step.

### Tuned settings

| Goal | Steps | Guidance | Time |
|---|---|---|---|
| Fast drafts | 20 | 1.0 | ~9.6 s |
| Normal use (recommended) | 20 | 3.0 | ~19 s |
| High detail | 30 | 3.0–4.0 | ~33 s |

Upstream recommends ≥50 steps. Measured here, 20 and 50 steps are
indistinguishable for anime subjects, so the default 50 is usually wasted time.
Guidance above 1.0 keeps colors richer; guidance 1.0 is visibly flatter.

`uncond_skip_layers` does **not** disable negative prompts. Negative prompts are
still encoded and still condition the unconditional branch; skip-layer guidance
just runs that branch with layer 22 (and any others listed) skipped, as a compute
saving. The two features compose.

---

## Quick start

The vendor PyTorch must be installed first. Installing a generic PyPI torch will
replace the DTK build and break the accelerator path.

```bash
# 1. Environment (reuse the vendor torch already present in the image)
python3 -m venv --system-site-packages /workspace/venv-moss
/workspace/venv-moss/bin/pip install -r requirements.txt

# 2. Weights (26 GB)
/workspace/venv-moss/bin/modelscope download \
  --model openmoss/MOSS-Image-SFT \
  --local_dir /workspace/models/MOSS-Image-SFT

# 3. Sanity checks — do not skip these
/workspace/venv-moss/bin/python tools/moss_te_check.py
/workspace/venv-moss/bin/python tools/moss_timestep_check.py

# 4. Generate
/workspace/venv-moss/bin/python tools/moss_infer.py \
  --prompt "masterpiece, anime illustration of a girl under cherry blossoms" \
  --steps 20 --guidance 3.0 --out /workspace/bench/out.png
```

Minimal inference, if you would rather not use the scripts:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/workspace/models/MOSS-Image-SFT",
    custom_pipeline="pipeline_moss_image",   # lives in the model repo, not diffusers
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    local_files_only=True,
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")

image = pipe(
    prompt,
    height=1024, width=1024,
    num_inference_steps=20,
    guidance_scale=3.0,
    uncond_skip_layers=[22],
    generator=torch.Generator("cpu").manual_seed(2024),
).images[0]
```

Notes that cost time to discover:

- `custom_pipeline="pipeline_moss_image"` is mandatory. The pipeline class ships
  inside the model repository.
- `torch_dtype` must be a dict. The VAE is fp32 by design (`force_upcast: true`).
- The tokenizer lives in `tokenizer/`, not under `text_encoder/`.
- Dimensions must be multiples of 16.

---

## The deadlock on ROCm/DTK

Worth reading if you plan to serve any diffusion model here.

**Symptom.** The web UI stops responding. The process is alive, `hy-smi` still
shows VRAM allocated, but nothing listens on the port and `curl` to localhost
fails from the remote box itself.

**Cause.** Two mistakes compounding:

1. The health endpoint called `torch.cuda.memory_allocated()` on every request.
   The frontend polls it. Each poll enters the HIP runtime from an HTTP handler
   thread. While another thread is running a generation, concurrent entry into
   the HIP runtime deadlocks.
2. The generation held a global lock for its full 25–50 s duration, so polling
   requests queued on that lock and threads piled up until the socket stopped
   accepting connections.

**Fix.** Keep GPU calls on one worker and read a cache everywhere else:

- health returns a plain dict copy and never touches torch
- VRAM is sampled into that dict only while the GPU lock is held
- the GPU lock is acquired non-blocking, so a busy GPU returns HTTP 429
  immediately instead of parking a request thread on torch
- the frontend skips polling while a generation is in flight

**Verification.** During a 24.6 s generation, 40 health polls at 1/s all returned
200 with zero timeouts. Health also answers instantly while a model is loading.

**Rule.** On ROCm/DTK, do not call torch or CUDA/HIP APIs from request threads.
Single-owner GPU thread, cached status for everyone else.

---

## The web UI

`tools/ui_server.py` serves a single page with both backends and switches between
them, because Qwen-Image 2.1 (30.2 GiB) and MOSS-Image (25.4 GiB) cannot be
resident together on a 64 GB card.

```bash
# On the remote box, inside a persistent session (background processes are
# reaped when the SSH session closes, so nohup is not enough)
python3 tools/ui_server.py

# From your machine, forward the port
python3 tools/remote_tunnel.py 'cd /workspace && exec python3 ui_server.py'
# then open http://127.0.0.1:8081
```

Configuration comes from the environment; see `.env.example`. Nothing about any
particular machine is baked into the repo.

`tools/remote_exec.py`, `remote_run.py`, `remote_tunnel.py`, and `remote_put.py`
read `MOSS_HOST` / `MOSS_PORT` / `MOSS_USER` / `MOSS_PASS_FILE`. Leave
`MOSS_PASS_FILE` unset to use keys or an agent, which is the recommended setup.

---

## Quality notes

- **Chinese handwritten text is a genuine strength.** The upstream refrigerator
  prompt (three lines of handwritten Chinese on a sticky note) renders every
  character correctly, in a natural handwriting style.
- **Anime is strong.** Detailed eyes, layered hair, cel shading. Compared with
  Qwen-Image 2.1 on an identical prompt and seed, MOSS produces a tighter
  close-up with more surface detail; Qwen produces a softer, lighter mid-shot.
- **bf16 residual stream.** Upstream runs everything in bf16, and `issue #31`
  measured 2.4–3.5× higher prediction error than an fp32 residual stream.
  Upstream declined the change to stay consistent with training. If output looks
  soft, casting non-`layers.*.proj.*` parameters to fp32 is a known mitigation
  costing about 1.5% speed. Not applied here.

See [docs/PROMPTS.md](docs/PROMPTS.md) for tested anime prompt recipes.

---

## Deliberately not done

- **No `causal_conv1d`.** Building it from source failed against this toolchain,
  and the only gain is the 6.6 s first text encode. Not worth the risk.
- **No `fla`.** An earlier `fla` build on this card produced zero forward and NaN
  backward. The pure-PyTorch path is verified correct; there is no reason to
  reintroduce a dependency known to fail here.
- **No fp32 residual-stream patch.** Default bf16 output is acceptable and
  matches upstream training.
- **No ComfyUI.** ComfyUI ships its own pure-PyTorch Qwen3.5 with GDN
  (`comfy/text_encoders/qwen35.py`), which is a viable alternative path, but
  unnecessary for this use case.

---

## Repository layout

```
tools/
  remote_config.py         shared SSH config from environment
  remote_exec.py           one-shot remote command
  remote_run.py            persistent remote session (for long jobs)
  remote_tunnel.py         persistent session + local port forwarding
  remote_put.py            upload a file over SSH
  ui_server.py             web UI, dual backend, model switching
  moss_infer.py            single generation with timing
  moss_te_check.py         text encoder numeric verification
  moss_timestep_check.py   float64 timestep verification
  moss_breakdown.py        latency breakdown: encode / denoise / VAE
  bench_bw.py              compute and bandwidth microbenchmarks
docs/
  DEPLOYMENT.md            full deployment log and troubleshooting
  PROMPTS.md               tested anime prompt recipes
```

## Licenses

This repository is MIT (see `LICENSE`). It contains no model weights.

- MOSS-Image: Apache 2.0, by OpenMOSS.
- Qwen-Image 2.1: Qwen Research License Agreement — **not** an open-source
  license. Referenced only as an optional comparison backend.

## Acknowledgements

Upstream work that made this possible: [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image),
[Hugging Face diffusers](https://github.com/huggingface/diffusers), and
[transformers](https://github.com/huggingface/transformers) — in particular its
graceful `fla` fallback, without which this would not have run unmodified.

Corrections welcome, especially from anyone running this on other DCU models,
ROCm cards, or with `fla`/`causal_conv1d` successfully built.
