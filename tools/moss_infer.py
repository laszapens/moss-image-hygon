"""MOSS-Image inference with staged timing and VRAM measurement."""
import argparse
import json
import os
import time

import torch
from diffusers import DiffusionPipeline

MODEL = "/workspace/models/MOSS-Image-SFT"


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A red apple on a wooden table, soft window light.")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out", default="/workspace/bench/moss/out.png")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--offload", action="store_true")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        local_files_only=True,
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    print(f"[load] from_pretrained {time.perf_counter()-t0:.1f}s "
          f"vram={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    if args.offload:
        pipe.enable_model_cpu_offload()
        print("[load] model cpu offload enabled", flush=True)
    else:
        t1 = time.perf_counter()
        pipe.to("cuda")
        torch.cuda.synchronize()
        print(f"[load] .to(cuda) {time.perf_counter()-t1:.1f}s "
              f"vram={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    print(f"[vram] after load alloc={torch.cuda.memory_allocated()/2**30:.2f} "
          f"peak={peak_gb():.2f}GiB", flush=True)

    skip = None if args.no_skip else [22]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        uncond_skip_layers=skip,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)

    print("[result] " + json.dumps({
        "resolution": f"{args.width}x{args.height}",
        "steps": args.steps,
        "guidance": args.guidance,
        "skip": skip,
        "offload": args.offload,
        "total_s": round(dt, 2),
        "per_step_s": round(dt / args.steps, 3),
        "peak_vram_GiB": round(peak_gb(), 2),
        "out": args.out,
    }), flush=True)


if __name__ == "__main__":
    main()
