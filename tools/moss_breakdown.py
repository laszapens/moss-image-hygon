"""Break down MOSS-Image latency: text encoding vs DiT denoising vs VAE decode."""
import json
import time

import torch
from diffusers import DiffusionPipeline

MODEL = "/workspace/models/MOSS-Image-SFT"
PROMPT = ("masterpiece, best quality, anime illustration of a young girl with long silver hair "
          "and blue eyes, wearing a white sailor uniform, standing under cherry blossoms")

pipe = DiffusionPipeline.from_pretrained(
    MODEL, custom_pipeline="pipeline_moss_image", trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    local_files_only=True,
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")
torch.cuda.synchronize()

print("=== component timings ===")

# 1) Text encoding (this is the 16GB Qwen3.5-9B, run on CPU-fallback GDN ops)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
emb, mask = pipe.encode_prompt(PROMPT, 4096)
torch.cuda.synchronize()
t_te = time.perf_counter() - t0
tok_len = int(mask.sum())
print(f"[text encoder] {t_te:6.2f}s  tokens={tok_len}  "
      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")

# Re-encode to get a hot-cache number (page cache warm, kernels compiled)
t0 = time.perf_counter()
emb, mask = pipe.encode_prompt(PROMPT, 4096)
torch.cuda.synchronize()
print(f"[text encoder] {time.perf_counter()-t0:6.2f}s  (second call, warm)")

# 2) Full generation, timed per-phase through the pipeline internals
for steps in (20, 50):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    img = pipe(PROMPT, height=1024, width=1024, num_inference_steps=steps,
               guidance_scale=3.0, uncond_skip_layers=[22],
               generator=torch.Generator("cpu").manual_seed(2024)).images[0]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    print(f"[full {steps:3d} steps] {total:6.2f}s  "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB  "
          f"per_step={(total - t_te)/steps:.3f}s (denoise only)")

print("\n=== math ===")
# MOSS runs 2 DiT forwards per step with guidance_scale != 1 (CFG), plus block-22
# skipping on the uncond branch. Qwen at cfg=1 runs 1 forward per step.
print("MOSS: 2 forwards/step (CFG) x N steps")
print("Qwen: 1 forward/step at cfg=1, 2 at cfg>1")
print("So MOSS-50steps ~= Qwen-100steps worth of DiT compute, plus a 9B text encode.")
