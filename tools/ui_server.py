"""Unified image-generation UI: Qwen-Image 2.1 + MOSS-Image on one Hygon BW card.

Both models are mutually exclusive in VRAM (30.2 + 25.4 GiB > 64 GiB with activations),
so exactly one is resident at a time and switching unloads the other.

Endpoints:
  GET  /                  -> page
  GET  /api/health        -> resident model, load state, vram
  POST /api/model         -> {model: "qwen"|"moss"|null} switch (null = unload all)
  POST /api/generate      -> {prompt, negative_prompt, width, height, steps, cfg, seed, ...}
  GET  /api/image/<id>    -> PNG
  GET  /api/gallery       -> recent generations
  GET  /api/presets       -> per-model defaults
"""
import gc
import io
import json
import os
import threading
import time
import traceback
import uuid

from flask import Flask, Response, jsonify, request, send_file

QWEN_MODEL = os.environ.get("QWEN_IMAGE_MODEL", "/workspace/models/Qwen-Image-2.1")
MOSS_MODEL = os.environ.get("MOSS_IMAGE_MODEL", "/workspace/models/MOSS-Image-SFT")
OUTDIR = os.environ.get("GI_OUT", "/workspace/bench/gallery")
PORT = int(os.environ.get("GI_PORT", "8081"))

app = Flask(__name__)

# Per-model recommended defaults, derived from measured sweeps on this card.
PRESETS = {
    "qwen": {
        "label": "Qwen-Image 2.1",
        "steps": 20,
        "cfg": 1.0,
        "cfg_hint": "1.0 最自然；>1 才启用负向词（代价翻倍）",
        "size": [1024, 1024],
        "notes": "强指令跟随、写实与文字都稳；cfg=1 时 20 步即饱和",
        "vram_gib": 30.2,
    },
    "moss": {
        "label": "MOSS-Image SFT",
        "steps": 20,
        "cfg": 3.0,
        "cfg_hint": "3.0 官方推荐（每步 2 次前向，约 19s）；改 1.0 可省一半时间但色彩变淡",
        "size": [1024, 1024],
        "notes": "二次元/中文手写文字强；SLG 跳层 22 默认开启",
        "vram_gib": 25.4,
    },
}

_state = {
    "loaded": None,      # "qwen" | "moss" | None
    "loading": None,     # model key while loading
    "busy": False,
    "error": None,
    "pipes": {},
    "load_s": {},
}
_lock = threading.Lock()
# Serialises actual CUDA work (load/generate). Kept separate from _lock so that
# cheap request handlers never block behind a 25-second generation.
_gpu_lock = threading.Lock()

# Cached VRAM snapshot. torch/HIP calls may only happen on the thread that owns
# the CUDA context; calling them from a request thread while another thread is
# generating deadlocks the HIP runtime (observed: process alive, port dead).
# Handlers read this dict only; _refresh_vram() writes it.
_vram_cache = {"alloc_gib": 0.0, "peak_gib": 0.0, "total_gib": 64.0, "device": "BW"}


def _refresh_vram():
    """Sample VRAM. Call ONLY while holding _gpu_lock."""
    try:
        import torch
        if torch.cuda.is_available():
            _vram_cache.update({
                "alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                "device": torch.cuda.get_device_name(0),
            })
    except Exception:
        pass
    return dict(_vram_cache)


def _vram():
    """Snapshot for request handlers: pure dict copy, never touches torch."""
    return dict(_vram_cache)


def _unload():
    """Free whatever is resident. Returns the key that was unloaded."""
    import torch
    prev = _state["loaded"]
    if prev and prev in _state["pipes"]:
        pipe = _state["pipes"].pop(prev, None)
        del pipe
        _state["loaded"] = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return prev


def _load_qwen():
    import torch
    from diffusers import QwenImage21Pipeline
    t0 = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    torch.cuda.synchronize()
    return pipe, time.perf_counter() - t0


def _load_moss():
    import torch
    from diffusers import DiffusionPipeline
    t0 = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        MOSS_MODEL,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        local_files_only=True,
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    torch.cuda.synchronize()
    return pipe, time.perf_counter() - t0


LOADERS = {"qwen": _load_qwen, "moss": _load_moss}


def _ensure_model(key):
    """Load `key`, unloading anything else first. Caller must hold _lock."""
    if _state["loaded"] == key:
        return
    if _state["loaded"]:
        print(f"[switch] unloading {_state['loaded']}", flush=True)
        _unload()
    _state["loading"] = key
    _state["error"] = None
    try:
        print(f"[switch] loading {key}", flush=True)
        pipe, dt = LOADERS[key]()
        _state["pipes"][key] = pipe
        _state["loaded"] = key
        _state["load_s"][key] = round(dt, 1)
        print(f"[switch] {key} ready in {dt:.1f}s", flush=True)
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _state["loading"] = None


def _generate(key, body):
    import torch

    pipe = _state["pipes"][key]
    width = body["width"]
    height = body["height"]
    steps = body["steps"]
    seed = body["seed"]
    gen = torch.Generator("cpu").manual_seed(seed)

    if key == "qwen":
        kw = dict(prompt=body["prompt"], height=height, width=width,
                  num_inference_steps=steps, generator=gen)
        cfg = body.get("cfg", 1.0)
        if cfg and cfg > 1.0:
            kw["true_cfg_scale"] = cfg
            if body.get("negative_prompt"):
                kw["negative_prompt"] = body["negative_prompt"]
    else:
        kw = dict(prompt=body["prompt"], height=height, width=width,
                  num_inference_steps=steps, generator=gen,
                  guidance_scale=body.get("cfg", 3.0))
        neg = body.get("negative_prompt")
        if neg:
            kw["negative_prompt"] = neg
        skip = body.get("uncond_skip")
        kw["uncond_skip_layers"] = None if skip in (None, [], "none") else skip

    return pipe(**kw).images[0]


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/presets")
def presets():
    return jsonify(PRESETS)


@app.get("/api/health")
def health():
    return jsonify({
        "loaded": _state["loaded"],
        "loading": _state["loading"],
        "busy": _state["busy"],
        "error": _state["error"],
        "load_s": _state["load_s"],
        "vram": _vram(),
    })


@app.post("/api/model")
def switch_model():
    body = request.get_json(force=True) or {}
    key = body.get("model")
    if key not in (None, "qwen", "moss"):
        return jsonify({"error": "model must be qwen, moss, or null"}), 400
    if _state["busy"] or _state["loading"]:
        return jsonify({"error": "busy", "detail": "another job is running"}), 429
    # Non-blocking acquire: a long generation must never queue up request threads.
    if not _gpu_lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "GPU is occupied"}), 429
    try:
        if key is None:
            _unload()
        else:
            _ensure_model(key)
        _refresh_vram()
    finally:
        _gpu_lock.release()
    if _state["error"]:
        return jsonify({"error": _state["error"]}), 500
    return jsonify({"loaded": _state["loaded"], "load_s": _state["load_s"],
                    "vram": _vram()})


@app.post("/api/generate")
def generate():
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    key = body.get("model") or _state["loaded"]
    if key not in PRESETS:
        return jsonify({"error": "no model selected", "detail": "call /api/model first"}), 400
    if _state["busy"]:
        return jsonify({"error": "busy", "detail": "another generation is running"}), 429

    try:
        width = max(256, min(2048, int(body.get("width", 1024)) // 16 * 16))
        height = max(256, min(2048, int(body.get("height", 1024)) // 16 * 16))
        steps = max(1, min(100, int(body.get("steps", PRESETS[key]["steps"]))))
        cfg = float(body.get("cfg", PRESETS[key]["cfg"]))
        seed = int(body.get("seed", -1))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"bad parameter: {e}"}), 400

    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)

    body.update(width=width, height=height, steps=steps, seed=seed, cfg=cfg)

    # Non-blocking acquire: while another generation holds the GPU, reject this
    # one immediately instead of parking a request thread on torch.
    if not _gpu_lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "GPU is occupied"}), 429

    _state["busy"] = True
    result = {}
    try:
        import torch
        _ensure_model(key)
        if _state["error"]:
            return jsonify({"error": _state["error"]}), 500

        t0 = time.perf_counter()
        image = _generate(key, body)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        img_id = uuid.uuid4().hex[:16]
        os.makedirs(OUTDIR, exist_ok=True)
        image.save(os.path.join(OUTDIR, f"{img_id}.png"))
        peak = _refresh_vram()
        meta = {
            "id": img_id,
            "model": key,
            "model_label": PRESETS[key]["label"],
            "prompt": prompt,
            "negative_prompt": body.get("negative_prompt") or "",
            "width": width, "height": height,
            "steps": steps, "cfg": cfg, "seed": seed,
            "seconds": round(dt, 2),
            "per_step_s": round(dt / steps, 3),
            "vram": peak,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(OUTDIR, f"{img_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        result.update(meta)
    except Exception as e:
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        _state["busy"] = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _gpu_lock.release()

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.get("/api/image/<img_id>")
def image(img_id):
    if not img_id.isalnum():
        return jsonify({"error": "bad id"}), 400
    path = os.path.join(OUTDIR, f"{img_id}.png")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.get("/api/gallery")
def gallery():
    items = []
    if os.path.isdir(OUTDIR):
        for name in sorted(os.listdir(OUTDIR)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(OUTDIR, name), encoding="utf-8") as f:
                        items.append(json.load(f))
                except Exception:
                    pass
    items.sort(key=lambda m: m.get("created", ""), reverse=True)
    return jsonify(items[:80])


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图像生成 · 海光 BW 64GB</title>
<style>
  :root {
    --bg:#0b0d10; --panel:#15181d; --panel2:#1c2027; --line:#2a3038;
    --fg:#e6e9ef; --dim:#8d97a6; --accent:#4f8cff; --ok:#3ddc84; --err:#ff5c5c; --warn:#ffb020;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
  .wrap{display:grid;grid-template-columns:370px 1fr;min-height:100vh}
  aside{background:var(--panel);border-right:1px solid var(--line);padding:18px;overflow-y:auto;max-height:100vh}
  main{padding:22px;overflow-y:auto;max-height:100vh}
  h1{font-size:15px;margin:0 0 4px;letter-spacing:.2px}
  .sub{color:var(--dim);font-size:12px;margin-bottom:14px}
  label{display:block;font-size:12px;color:var(--dim);margin:12px 0 5px}
  textarea,input,select{width:100%;background:var(--panel2);color:var(--fg);border:1px solid var(--line);
    border-radius:7px;padding:9px 10px;font:inherit;font-size:13px;outline:none}
  textarea{resize:vertical;min-height:92px;line-height:1.55}
  textarea:focus,input:focus,select:focus{border-color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  button{width:100%;margin-top:16px;padding:11px;border:0;border-radius:8px;background:var(--accent);
    color:#fff;font:inherit;font-weight:600;cursor:pointer;transition:filter .15s}
  button:hover:not(:disabled){filter:brightness(1.12)}
  button:disabled{opacity:.5;cursor:not-allowed}
  button.ghost{background:var(--panel2);border:1px solid var(--line);color:var(--fg);margin-top:8px}
  .status{margin-top:14px;padding:10px 11px;border-radius:7px;background:var(--panel2);
    border:1px solid var(--line);font-size:12px;color:var(--dim);line-height:1.7}
  .status b{color:var(--fg);font-weight:600}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px}
  .dot.ok{background:var(--ok)} .dot.busy{background:var(--warn)} .dot.off{background:var(--err)}
  .err{color:var(--err)}
  .hintline{font-size:11px;color:var(--dim);margin-top:4px;line-height:1.5}
  .models{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:4px}
  .mbtn{padding:9px 8px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);
    color:var(--dim);font:inherit;font-size:12px;cursor:pointer;text-align:center;margin:0;transition:.15s}
  .mbtn:hover{border-color:var(--accent);color:var(--fg)}
  .mbtn.on{border-color:var(--accent);background:rgba(79,140,255,.13);color:var(--fg);font-weight:600}
  .stage{display:flex;align-items:center;justify-content:center;background:var(--panel);
    border:1px solid var(--line);border-radius:12px;min-height:420px;padding:16px;position:relative}
  .stage img{max-width:100%;max-height:72vh;border-radius:8px;display:block}
  .hint{color:var(--dim);text-align:center;font-size:13px;line-height:2}
  .meta{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12px;color:var(--dim)}
  .meta b{color:var(--fg);font-weight:600}
  .gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:9px;margin-top:18px}
  .gal figure{margin:0;position:relative}
  .gal img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:7px;border:1px solid var(--line);
    cursor:pointer;display:block}
  .gal img:hover{border-color:var(--accent)}
  .gal figcaption{position:absolute;left:4px;top:4px;background:rgba(0,0,0,.65);color:#fff;
    font-size:10px;padding:1px 5px;border-radius:4px;pointer-events:none}
  h2{font-size:12px;color:var(--dim);font-weight:600;margin:20px 0 0;letter-spacing:.4px}
  .spin{width:15px;height:15px;border:2px solid var(--line);border-top-color:var(--accent);
    border-radius:50%;animation:r .8s linear infinite;display:inline-block;vertical-align:-3px}
  @keyframes r{to{transform:rotate(360deg)}}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}aside{max-height:none}}
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>图像生成</h1>
    <div class="sub">海光 BW · 1×64GB HBM · 单卡串行</div>

    <label>模型</label>
    <div class="models">
      <button class="mbtn" id="m-qwen" data-model="qwen">Qwen-Image 2.1<br><span style="font-size:10px">写实 / 通用</span></button>
      <button class="mbtn" id="m-moss" data-model="moss">MOSS-Image<br><span style="font-size:10px">二次元 / 中文文字</span></button>
    </div>
    <div class="hintline" id="modelnote">切换模型会卸载当前模型（显存只够一个）</div>

    <label>提示词</label>
    <textarea id="prompt" placeholder="描述你想要的画面…">masterpiece, best quality, anime illustration of a young girl with long silver hair and blue eyes, wearing a white sailor uniform, standing under cherry blossoms, petals falling, soft sunlight, detailed eyes, cel shading, clean lineart</textarea>

    <label>负向提示词</label>
    <textarea id="negative" style="min-height:54px" placeholder="nsfw, worst quality, bad anatomy, watermark">nsfw, worst quality, bad anatomy, watermark</textarea>

    <div class="row">
      <div><label>宽</label>
        <select id="width"><option>768</option><option>1024</option><option>1328</option>
        <option>1536</option><option>2048</option></select></div>
      <div><label>高</label>
        <select id="height"><option>768</option><option selected>1024</option><option>1328</option>
        <option>1536</option><option>2048</option></select></div>
    </div>

    <div class="row3">
      <div><label>步数</label><input id="steps" type="number" value="20" min="1" max="100"></div>
      <div><label id="cfglabel">CFG</label><input id="cfg" type="number" value="3.0" step="0.1" min="1" max="10"></div>
      <div><label>种子</label><input id="seed" type="number" value="-1" min="-1"></div>
    </div>
    <div class="hintline" id="cfghint"></div>

    <button id="go">生成</button>
    <button class="ghost" id="free">卸载模型释放显存</button>

    <div class="status" id="status"><span class="dot off"></span>连接中…</div>
  </aside>

  <main>
    <div class="stage" id="stage">
      <div class="hint">选模型 → 写提示词 → 生成<br>
      <span style="font-size:12px">首次加载约 15–25 秒，之后常驻不重复加载</span></div>
    </div>
    <div class="meta" id="meta"></div>
    <h2>最近生成</h2>
    <div class="gal" id="gal"></div>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
let PRESETS = {}, current = null, ready = false;
let generating = false, lastOk = Date.now();

async function loadPresets() {
  PRESETS = await (await fetch('/api/presets')).json();
}

function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  $('steps').value = p.steps;
  $('cfg').value = p.cfg;
  $('cfghint').textContent = p.cfg_hint || '';
  $('cfglabel').textContent = key === 'moss' ? 'Guidance' : 'CFG';
  $('width').value = p.size[0];
  $('height').value = p.size[1];
}

async function selectModel(key, silent) {
  if (!silent && current === key) return;
  current = key;
  document.querySelectorAll('.mbtn').forEach(b =>
    b.classList.toggle('on', b.dataset.model === key));
  applyPreset(key);
  $('modelnote').textContent = PRESETS[key].notes;
  if (silent) return;
  $('go').disabled = true;
  $('status').innerHTML = '<span class="spin"></span> 切换模型中…（会卸载旧模型）';
  try {
    const r = await fetch('/api/model', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: key}),
    });
    const m = await r.json();
    if (!r.ok || m.error) throw new Error(m.error || ('HTTP ' + r.status));
  } catch (e) {
    $('status').innerHTML = '<span class="dot off"></span><span class="err">' + e.message + '</span>';
  } finally {
    $('go').disabled = false;
    poll();
  }
}

async function freeAll() {
  $('go').disabled = true;
  $('status').innerHTML = '<span class="spin"></span> 卸载中…';
  try { await fetch('/api/model', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({model: null})}); } catch(e) {}
  $('go').disabled = false;
  poll();
}

function setStatus(h) {
  const v = h.vram || {};
  if (h.error) {
    $('status').innerHTML = '<span class="dot off"></span><span class="err">出错</span><br>' + h.error;
  } else if (h.loading) {
    $('status').innerHTML = '<span class="spin"></span> 加载 ' + (PRESETS[h.loading]?.label || h.loading) + '…';
  } else if (h.busy) {
    $('status').innerHTML = '<span class="dot busy"></span><b>生成中…</b>';
  } else if (h.loaded) {
    $('status').innerHTML = '<span class="dot ok"></span><b>' + (PRESETS[h.loaded]?.label || h.loaded) + '</b> 已就绪<br>' +
      '显存 <b>' + (v.alloc_gib ?? '?') + ' / ' + (v.total_gib ?? '?') + ' GiB</b>' +
      (h.load_s && h.load_s[h.loaded] ? ' · 加载 ' + h.load_s[h.loaded] + 's' : '');
  } else {
    $('status').innerHTML = '<span class="dot off"></span>无模型常驻 · 生成时自动加载';
  }
}

async function poll() {
  // Skip while a generation is in flight: the server answers 429 anyway, and
  // piling up requests during a 25s job is what wedged the old build.
  if (generating) return;
  try {
    const h = await (await fetch('/api/health')).json();
    ready = !!h.loaded;
    if (h.loaded && current !== h.loaded) selectModel(h.loaded, true);
    if (!current && !h.loaded) selectModel('qwen', true);
    setStatus(h);
    lastOk = Date.now();
  } catch (e) {
    if (Date.now() - lastOk > 8000) {
      $('status').innerHTML = '<span class="dot off"></span><span class="err">后端无响应</span>';
    }
  }
}

async function generate() {
  const prompt = $('prompt').value.trim();
  if (!prompt) { alert('请先输入提示词'); return; }
  if (!current) { alert('请先选择模型'); return; }
  generating = true;
  $('go').disabled = true;
  $('stage').innerHTML = '<div class="hint"><span class="spin"></span><br>生成中，请稍候…</div>';
  $('meta').innerHTML = '';
  const body = {
    model: current, prompt,
    negative_prompt: $('negative').value.trim(),
    width: +$('width').value, height: +$('height').value,
    steps: +$('steps').value, cfg: +$('cfg').value, seed: +$('seed').value,
  };
  const t0 = performance.now();
  try {
    const r = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body),
    });
    const m = await r.json();
    if (!r.ok || m.error) throw new Error(m.error || ('HTTP ' + r.status));
    const img = new Image();
    img.onload = () => { $('stage').innerHTML = ''; $('stage').appendChild(img); };
    img.src = '/api/image/' + m.id + '?t=' + Date.now();
    $('seed').value = m.seed;
    $('meta').innerHTML =
      '<span>模型 <b>' + m.model_label + '</b></span>' +
      '<span>尺寸 <b>' + m.width + '×' + m.height + '</b></span>' +
      '<span>步数 <b>' + m.steps + '</b></span>' +
      '<span>CFG <b>' + m.cfg + '</b></span>' +
      '<span>种子 <b>' + m.seed + '</b></span>' +
      '<span>耗时 <b>' + m.seconds + 's</b></span>' +
      '<span>每步 <b>' + m.per_step_s + 's</b></span>' +
      '<span>峰值显存 <b>' + (m.vram && m.vram.peak_gib) + ' GiB</b></span>' +
      '<span>端到端 <b>' + ((performance.now()-t0)/1000).toFixed(1) + 's</b></span>';
    loadGallery();
  } catch (e) {
    $('stage').innerHTML = '<div class="hint err">生成失败<br>' + e.message + '</div>';
  } finally {
    generating = false;
    $('go').disabled = false;
    poll();
  }
}

async function loadGallery() {
  try {
    const items = await (await fetch('/api/gallery')).json();
    $('gal').innerHTML = items.map(m =>
      '<figure><img src="/api/image/' + m.id + '" title="' +
      (m.prompt||'').replace(/"/g,'&quot;') + '&#10;' + m.width + '×' + m.height +
      ' · ' + m.steps + ' steps · ' + m.seconds + 's" onclick="reuse(\'' + m.id + '\')">' +
      '<figcaption>' + (m.model === 'moss' ? 'MOSS' : 'Qwen') + '</figcaption></figure>'
    ).join('');
  } catch (e) {}
}

async function reuse(id) {
  const items = await (await fetch('/api/gallery')).json();
  const m = items.find(x => x.id === id);
  if (!m) return;
  if (m.model && m.model !== current) selectModel(m.model, true);
  $('prompt').value = m.prompt || '';
  $('negative').value = m.negative_prompt || '';
  $('width').value = m.width; $('height').value = m.height;
  $('steps').value = m.steps; $('cfg').value = m.cfg; $('seed').value = m.seed;
  const img = new Image();
  img.onload = () => { $('stage').innerHTML = ''; $('stage').appendChild(img); };
  img.src = '/api/image/' + m.id;
  $('meta').innerHTML = '<span>已载入历史参数，可改提示词重新生成</span>';
}

document.querySelectorAll('.mbtn').forEach(b =>
  b.onclick = () => selectModel(b.dataset.model));
$('go').onclick = generate;
$('free').onclick = freeAll;
$('prompt').addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') generate();
});

(async () => {
  await loadPresets();
  selectModel('qwen', true);
  poll();
  loadGallery();
  setInterval(poll, 4000);
})();
</script>
</body>
</html>
"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # Do not preload: the user picks a model, and switching costs one reload.
    print(f"[server] serving on 127.0.0.1:{PORT}", flush=True)
    print(f"[server] qwen={QWEN_MODEL}", flush=True)
    print(f"[server] moss={MOSS_MODEL}", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
