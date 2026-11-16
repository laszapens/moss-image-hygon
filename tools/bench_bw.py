import torch, time

def bench_matmul(dtype, n=8192, iters=30, warmup=8):
    torch.cuda.empty_cache()
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * n**3
    tflops = flops / dt / 1e12
    print(f"matmul {str(dtype):>14} n={n}: {dt*1e3:8.2f} ms  {tflops:7.2f} TFLOP/s")
    del a, b, c
    torch.cuda.empty_cache()
    return tflops

def bench_bw(iters=60, warmup=10):
    n = 1 << 28  # 1GiB of fp32 = 268M elems
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    for _ in range(warmup):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    gb = 2 * x.numel() * 4 / 1e9  # read + write
    bw = gb / dt
    print(f"copy bandwidth (r+w): {bw:8.1f} GB/s   ({dt*1e3:.2f} ms per 1GiB copy)")
    del x, y
    torch.cuda.empty_cache()
    return bw

def bench_attn(dtype=torch.bfloat16, seq=4096, heads=24, dim=128, iters=10, warmup=4):
    try:
        q = torch.randn(1, heads, seq, dim, dtype=dtype, device="cuda")
        torch.cuda.synchronize()
        for _ in range(warmup):
            o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        flops = 4 * heads * seq * seq * dim
        print(f"sdpa bf16 seq={seq} h={heads} d={dim}: {dt*1e3:8.2f} ms  {flops/dt/1e12:7.2f} TFLOP/s")
        del q, o
        torch.cuda.empty_cache()
    except Exception as e:
        print("sdpa failed:", type(e).__name__, str(e)[:200])

print("=== device ===")
p = torch.cuda.get_device_properties(0)
print(p.name, "CUs", p.multi_processor_count)
print()
print("=== compute ===")
bench_matmul(torch.bfloat16)
bench_matmul(torch.float16)
bench_matmul(torch.float32, n=4096)
print()
print("=== memory ===")
bench_bw()
print()
print("=== attention ===")
bench_attn()
