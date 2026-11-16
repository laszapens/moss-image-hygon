"""Verify the Qwen3.5 text encoder produces sane embeddings on this DCU.

This is the make-or-break check for MOSS-Image on gfx936: the text encoder is
Qwen3_5Model with GDN (gated delta rule) linear attention. FLA kernels returned
zeros on this card in an earlier project; transformers 5.23 defaults to a pure
PyTorch fallback, and this script proves which path ran and whether it is sane.
"""
import torch
from transformers import AutoModel, AutoTokenizer

TE = "/workspace/models/MOSS-Image-SFT/text_encoder"
TOK = "/workspace/models/MOSS-Image-SFT/tokenizer"


def stats(name, t):
    tf = t.float()
    print(
        f"  {name:<22} shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={tf.mean():+.5f} std={tf.std():.5f} "
        f"min={tf.min():+.4f} max={tf.max():+.4f} "
        f"nan={torch.isnan(tf).any().item()} inf={torch.isinf(tf).any().item()} "
        f"zeros%={(tf == 0).float().mean() * 100:.2f}"
    )


def main():
    print("=== device ===")
    print(torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()

    print("\n=== which GDN path? ===")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    for fn in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        f = getattr(m, fn, None)
        print(f"  {fn}: {getattr(f, '__name__', f)}  module={getattr(f, '__module__', '?')}")
    try:
        import fla  # noqa: F401
        print("  fla: INSTALLED")
    except ImportError:
        print("  fla: not installed -> pure torch path")

    print("\n=== load tokenizer + model ===")
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    model = AutoModel.from_pretrained(TE, dtype=torch.bfloat16, local_files_only=True).eval().to("cuda")
    torch.cuda.synchronize()
    print(f"  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B "
          f"vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")

    prompts = [
        "A wide view of a refrigerator beside a kitchen counter with a fruit bowl.",
        "一张贴在冰箱门上的便签，手写着：记得买牛奶和鸡蛋。",
        "a red apple",  # very short, exercises different sequence lengths
    ]

    for p in prompts:
        print(f"\n--- prompt: {p[:50]}")
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tok(formatted, return_tensors="pt").to("cuda")
        print(f"  tokens={inputs['input_ids'].shape[1]}")
        with torch.no_grad():
            out = model(**inputs, use_cache=False, return_dict=True)
        h = out.last_hidden_state
        stats("last_hidden_state", h)
        stats("last token", h[0, -1])
        # A broken GDN path typically yields constant or zero vectors per position.
        per_pos_std = h[0].float().std(dim=0)
        print(f"  per-position std across dim: min={per_pos_std.min():.6f} max={per_pos_std.max():.6f}")
        if h.float().std() < 1e-3:
            print("  *** SUSPECT: embedding variance is near zero")
        # Check that different prompts give different embeddings (no collapse).
        globals().setdefault("_embs", []).append(h[0, -1].float().cpu())

    embs = globals().get("_embs", [])
    if len(embs) >= 2:
        print("\n=== cross-prompt separation (last token, cosine) ===")
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                a, b = embs[i], embs[j]
                cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                print(f"  pair({i},{j}) cos={cos:+.4f}")
        print("  (identical/collapsed embeddings would approach 1.0000)")

    print(f"\npeak vram: {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")


if __name__ == "__main__":
    main()
