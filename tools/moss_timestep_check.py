"""Verify the fp64 timestep sinusoid is not silently collapsing, and probe quality levers.

Research finding: on MPS/NPU the pipeline's float64 timestep embedding can silently
return zeros, producing pure noise with no error. This script proves what happens here.
"""
import torch

# Reproduce the pipeline's timestep embedding exactly as shipped.
def shipped(times, dim=256):
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


print("=== fp64 timestep sinusoid check ===")
ts = torch.tensor([1000.0, 976.0, 500.0, 24.0, 0.0], dtype=torch.float32, device="cuda")
emb = shipped(ts)
print(f"input times: {ts.tolist()}")
print(f"emb shape={tuple(emb.shape)} dtype={emb.dtype}")
print(f"emb std={emb.std():.6f} zeros%={(emb == 0).float().mean()*100:.2f} "
      f"nan={torch.isnan(emb).any().item()}")
# Each timestep must produce a distinct embedding.
norms = emb.norm(dim=1)
print(f"per-timestep L2 norm: {[round(v,3) for v in norms.tolist()]}")
if (norms == 0).any():
    print("*** FAIL: some timesteps produced all-zero embeddings (MPS/NPU bug)")
elif emb.std() < 1e-6:
    print("*** FAIL: embedding is constant")
else:
    print("OK: timesteps produce distinct non-zero embeddings")

# Compare against a pure-fp32 computation to see if fp64 matters here.
def f32(times, dim=256):
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float32) / half)
    phase = times.float()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()

e32 = f32(ts)
diff = (emb - e32).abs()
print(f"\nfp64 vs fp32 path: max_abs_diff={diff.max():.3e} mean={diff.mean():.3e}")
print("(large diff would mean fp32 would change results materially)")


print("\n=== fp32 residual-stream probe (issue #31) ===")
print("Checking whether block linear layers can be selectively kept bf16 while")
print("casting the residual stream to fp32.")
try:
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "/workspace/models/MOSS-Image-SFT",
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        local_files_only=True,
    )
    tr = pipe.transformer
    n_proj = sum(1 for n, _ in tr.named_parameters()
                 if n.startswith("layers.") and ".proj." in n)
    n_other = sum(1 for n, _ in tr.named_parameters()
                  if not (n.startswith("layers.") and ".proj." in n))
    print(f"layers.*.proj.* params: {n_proj} (stay bf16)")
    print(f"other params:           {n_other} (candidate for fp32)")
    print(f"transformer num_layers={tr.config.num_layers}")
    # Inspect one modulation to see dtype
    for name, p in list(tr.named_parameters())[:6]:
        print(f"  {name:<50} {p.dtype}")
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:300])
