# MOSS-Image 部署记录（海光 BW 64GB）

> 部署日期：2026-11-15  
> 状态：**已跑通，无官方 DCU 适配，属首次公开验证**  
> 附带双后端 Web UI（`tools/ui_server.py`，端口 8081）

---

## 0. 一句话结论

MOSS-Image 在这张卡上**开箱即通，不需要改一行模型代码**。纯 PyTorch 路径自动接管了 Qwen3.5 的 GDN 线性注意力，fp64 时间步也没有触发 MPS/NPU 那个静默归零 bug。1024×1024 / 20 步：guidance=3.0 时 **19.8 秒**，guidance=1.0 时 **9.6 秒**，峰值显存 28.2 GiB。

---

## 1. 先回答"为什么比 Qwen 慢"

"10GB 级"只说了 DiT。完整管线是 26GB：

| 组件 | 磁盘 | 常驻显存 |
|---|---|---|
| transformer (DiT, 42 层 dim3072) | **9.5 GB** | — |
| text_encoder (Qwen3.5-9B, 8.39B 参数) | **16 GB** | — |
| vae (FLUX.2 small) | 238 MB | — |
| **合计** | **26 GB** | **25.4 GiB**（载入后）/ 28.2 GiB（生成峰值） |

官方 README 那个"3.65 GiB"是开了 hybrid offload（DiT 留 CPU、其余从磁盘按需加载）的结果，代价是每次都要搬权重。

**两个真实的耗时来源（实测）：**

1. **文本编码器首次推理 6.64 秒**，第二次 0.10 秒。因为 `fla` 和 `causal_conv1d` 没装，transformers 走纯 PyTorch 参考实现（日志会明说 `falling back ... much slower`）。
2. **每步跑两次前向。** `guidance_scale != 1` 即 CFG，条件与无条件分支各一次。Qwen 在 cfg=1 时每步只跑一次。

对比（都是 1024×1024）：

| | 步数 | 耗时 | 每步 |
|---|---|---|---|
| Qwen | 20 | 8.2 s | 0.40 s |
| MOSS (guidance 3.0) | 20 | 19.8 s | 0.66 s |
| MOSS (guidance 1.0) | 20 | 9.6 s | 0.48 s |
| MOSS (guidance 3.0) | 50 | 45.9 s | 0.79 s |

MOSS 跑 50 步 ≈ Qwen 跑 100 步的 DiT 计算量。不是硬件问题，是默认参数 + 缺 kernel。

---

## 2. 兼容性核查（关键结论）

拓荒前我派子代理把 GitHub issues、HF discussions、官方文档、transformers 源码全查了一遍。结论按风险排序：

| 风险点 | 实际结果 |
|---|---|
| GDN 线性注意力无 kernel | ✅ **安全**。transformers 5.23 用 `use_kernel_func_from_hub_with_fallback`，纯 torch 是默认路径 |
| fp64 时间步静默归零（MPS/NPU 的坑） | ✅ **没踩到**。实测每个时间步 L2 范数都是 11.314 |
| DiT 自实现，可能依赖 CUDA 算子 | ✅ **安全**。DiT 只用标准 `F.scaled_dot_product_attention` |
| torch 2.7.1 低于官方声明 2.9 | ⚠️ 声明不符但**实测无影响** |
| 无任何 ROCm/DCU 前人报告 | ✅ 我们**是第一个**跑通的 |

**官方对非 NVIDIA 的态度：** MOSS 官方支持 MPS（Apple Silicon，有 `docs/macos.md`）和 Ascend NPU（PR #15），但**没有任何 ROCm/AMD/海光适配**。所以这次确实是拓荒。

**为什么能一次成功：** 这个 pipeline 写得很干净——DiT 自包含在 `pipeline_moss_image.py` 里，注意力用标准 SDPA，不碰 Triton 自定义 kernel。唯一的外部依赖是 transformers 的 Qwen3.5，而它自带纯 torch 降级路径。

### 2.1 两个必须知道的官方坑（我们没踩，但你换环境要留意）

1. **fp64 时间步**（issue #5 / #23）：`freq = 10000 ** (-arange(dtype=float64))` + `times.double()`。在不支持 fp64 的后端上会静默返回全零，导致**出纯噪声且不报错**。我们的卡支持 fp64，正常。如果以后换到别的不支持 fp64 的卡，要改成 CPU-fp64 再转回。
2. **bf16 残差流精度**（issue #31）：官方用 bf16 训练，推理也全 bf16，42 层每层都舍入。有人实测把残差流转 fp32 能把误差降 2.4–3.5 倍，代价 +1.5% 时间。官方拒绝改（保持与训练一致）。如果觉得画质偏糊，可以试这个补丁。

---

## 3. 部署步骤（可复现）

环境复用 Qwen 那套 `/workspace/venv-moss`，**不需要新 venv**（diffusers 0.43.0 + transformers 5.23.0 已满足 ≥0.38 / ≥5.3）。

```bash
# 1. 下载（ModelScope 仓库名是小写 openmoss）
/workspace/venv-moss/bin/modelscope download \
  --model openmoss/MOSS-Image-SFT \
  --local_dir /workspace/models/MOSS-Image-SFT
# 26GB / 37 文件，约 6.5 分钟

# 2. 直接推理，无需任何 patch
```

最小推理代码：

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/workspace/models/MOSS-Image-SFT",
    custom_pipeline="pipeline_moss_image",   # 注意：pipeline 在模型仓库里
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
    guidance_scale=3.0,          # 官推 3.0–6.0
    uncond_skip_layers=[22],     # SLG 跳层，官推 22/24/26
    generator=torch.Generator("cpu").manual_seed(2024),
).images[0]
```

**关键细节：**

- `custom_pipeline="pipeline_moss_image"` 是必须的。pipeline 文件在模型仓库根目录，不在 diffusers 里。
- `torch_dtype` 必须传字典：`default` bf16，`vae` fp32。VAE 强制 fp32 是官方设计（`force_upcast: true`）。
- tokenizer 在 `tokenizer/` 子目录，不在 `text_encoder/` 下。用 `from_pretrained` 走整个目录时会自动处理；手动加载时要指对路径。
- `uncond_skip_layers` 不是"禁用负向词"。负向词仍会被编码并作用于无条件分支，SLG 只是让那个分支跳过第 22 层来省算力。两者可以同时用。
- 分辨率必须是 16 的倍数。推荐 1M / 1.5M / 2M 总像素。

---

## 4. 实测性能

### 4.1 分辨率 × 步数（guidance 3.0）

| 分辨率 | 步数 | 耗时 | 每步 | 峰值显存 |
|---|---|---|---|---|
| 1024×1024 | 20 | 19.8 s | 0.66 s | 28.2 GiB |
| 1024×1024 | 50 | 45.9 s | 0.79 s | 28.2 GiB |
| 1328×1328 | 20 | 37.6 s | 1.88 s | 31.3 GiB |
| 1536×1024 | 20 | 33.3 s | 1.67 s | 30.2 GiB |

模型载入 25.4 GiB，加载耗时 8–20 秒（页缓存热度影响）。

### 4.2 guidance 的代价与收益（重要）

| guidance | 每步前向 | 20 步耗时 | 画质 |
|---|---|---|---|
| 1.0 | 1 次 | **9.6 s** | 干净协调，但色彩偏淡、氛围减弱 |
| 3.0 | 2 次 | 18.9 s | 更贴提示词、色彩更艳，但有杂乱感 |

**这是 MOSS 上最有价值的优化**：guidance=1.0 直接砍掉一半时间。代价是失去引导，画面更"平淡"但结构完整、不崩坏。追求速度就设 1.0。

### 4.3 步数对画质的影响

同 seed（2024）对比：20 / 30 / 50 步的二次元图**肉眼几乎无差别**，30 步（33.3s）与 50 步（51.0s）质量相当。官方说"≥50 步最佳"，但实测 20–30 步已经够用，不必按默认 50 步跑。

---

## 5. 画质实测：二次元与中文文字

### 5.1 二次元（MOSS vs Qwen，同提示词同种子）

提示词：银白长发蓝眼少女、白色水手服、樱花树下、赛璐璐上色。seed=2024。

| | MOSS | Qwen-Image 2.1 |
|---|---|---|
| 速度 | 51.0 s（50步） | 14.5 s（20步） |
| 构图 | 近景大特写 | 中景半身 |
| 画风 | 精致细腻、高对比、眼睛高光层次强 | 柔和清新、淡粉调、通透 |
| 细节 | 发丝/蝴蝶发饰/缎带丰富 | 线条轻、细节较平 |
| Laplacian | 402.6 | 1189.9 |

关于最后一个指标：**Qwen 数值高不代表更清晰**。MOSS 这张是极浅景深特写，背景樱花全虚化，拉低了全图方差；Qwen 是全画面都有细节的中景。指标被构图差异污染了，不能横比。

肉眼看：MOSS 更接近"高质量插画"，Qwen 更接近"轻小说封面"。MOSS 在二次元的细腻度和厚重感上确实更强。

### 5.2 中文手写文字（MOSS 的杀手级能力）

用官方那个冰箱便签提示词（三行中文手写），MOSS 输出：

> 记得买牛奶和鸡蛋！
> 牛奶两盒，鸡蛋一打。
> 顺便带一袋面包。

**逐字全对**，而且是自然的中文手写笔触，不是印刷体。这个能力 Qwen-Image 2.1 做不到。视觉上很接近真实照片。

英文同理：写有 "MOSS IMAGE" 的牌子，20 步就渲染得清晰正确。

**结论：要中文文字、手写体、二次元，选 MOSS；要通用写实、要快，选 Qwen。**

---

## 6. 统一 UI（两个模型共用）

### 6.1 为什么必须切换

| 模型 | 常驻显存 |
|---|---|
| Qwen-Image 2.1 | 30.2 GiB |
| MOSS-Image | 25.4 GiB |

加起来 55.6 GiB，加上激活值必然 OOM。所以**一次只能常驻一个**，切换时先卸载旧的再加载新的。

### 6.2 架构

```
本地浏览器 127.0.0.1:8081
   ↑ SSH -L 8081（tools/remote_tunnel.py 长驻会话）
远端 127.0.0.1:8081  flask（tools/ui_server.py，按需加载/卸载）
```

### 6.3 启动

```bash
# 远端（长驻会话内运行；SSH 断开后后台进程会被平台清理）
python3 tools/ui_server.py

# 本机（转发端口）
python3 tools/remote_tunnel.py 'cd /workspace && exec python3 tools/ui_server.py'
```

启动时**不预加载任何模型**（显存 0），第一次生成或点模型按钮时才加载。

### 6.4 功能

- 顶部两个模型按钮，点击切换（自动卸载旧模型）。状态栏显示当前常驻模型与显存。
- 切换模型会自动套用该模型的推荐参数：Qwen → 20 步 / CFG 1.0；MOSS → 20 步 / Guidance 3.0。
- 参数标签随模型变（Qwen 叫 CFG，MOSS 叫 Guidance），下方有实时说明。
- 「卸载模型释放显存」按钮：把显存归零，给别的工作让路。
- 图库每张图带 MOSS/Qwen 角标，点击可载入那次的所有参数（包括模型）。
- 种子 -1 随机，出图后回填。

### 6.5 API

```bash
# 切换模型
curl -X POST http://127.0.0.1:8081/api/model \
  -H 'Content-Type: application/json' -d '{"model":"moss"}'

# 卸载
curl -X POST http://127.0.0.1:8081/api/model \
  -H 'Content-Type: application/json' -d '{"model":null}'

# 生成
curl -X POST http://127.0.0.1:8081/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"moss","prompt":"anime girl","width":1024,"height":1024,
       "steps":20,"cfg":3.0,"seed":42}'
```

`cfg` 字段对两个模型含义不同：Qwen 是 `true_cfg_scale`，MOSS 是 `guidance_scale`。

---

## 7. 故障手册

| 症状 | 原因 | 修复 |
|---|---|---|
| 加载报 `Qwen3_5Model` 相关错误 | transformers < 5.3 | 升级到 5.23.0 |
| `pipeline_moss_image` 找不到 | 没传 `custom_pipeline` | 加上 `custom_pipeline="pipeline_moss_image"` |
| 出纯噪声、不报错 | fp64 时间步归零（见 issue #5） | 改 CPU-fp64 算再转回（本卡不需要） |
| `sdpa adopt the new interface of flash-attn` | torch 走 cutlass 后端 | 可忽略 |
| `Launch params ... launch bounds` (rocblas) | rocBLAS 编译提示 | 可忽略，不影响结果 |
| `causal_conv1d` / `flash-linear-attention` falling back | 未安装，走纯 torch | 可忽略；想提速再单独编译 |
| 画质偏糊 | bf16 残差流（issue #31） | 试把非 `layers.*.proj.*` 参数转 fp32 |
| MOSS 比 Qwen 慢一倍 | guidance>1 每步两次前向 | 设 guidance=1.0 |
| 想用负向词但没生效 | guidance ≤ 1 时无 CFG | 提到 >1 |
| 切换模型卡住 | 上一个生成未结束 | `/api/health` 看 busy，等它完成 |
| SSH 断开 UI 消失 | 平台清理后台进程 | 用 `tools/remote_tunnel.py` 长驻 |
| **进程活着但端口不响应** | **见 11 节：HIP 多线程死锁（已修复）** | 用当前版本的 `tools/ui_server.py` |

---

## 11. 已修复的严重 bug：服务僵死

**现象：** UI 用一段时间后打不开，`/api/health` 超时。但远端进程还在、`hy-smi` 显示显存仍占着、端口 8081 却没人监听。远程 `curl 127.0.0.1:8081` 同样失败。

**根因（两个因素叠加）：**

1. **`_vram()` 在请求线程里调用 torch。** 原实现每次 `/api/health` 都执行 `torch.cuda.memory_allocated()`。前端每 2.5 秒轮询一次，双标签页就是每 1.25 秒两次。当另一个线程正在跑生成时，多个线程并发进入 HIP runtime → 死锁。DTK/ROCm 上尤其容易触发。
2. **生成任务持有全局 `_lock` 长达 25–50 秒。** 期间所有 health 请求排队等锁，线程不断堆积，最终 socket 不再接受新连接。

两者一起造成"进程活着、显存还在、服务已死"。

**修复：**

- `_vram()` 改为**纯字典拷贝**，绝不接触 torch。真正的采样移到 `_refresh_vram()`，只在持有 GPU 锁时调用。
- 新增独立的 `_gpu_lock`，**非阻塞获取**（`acquire(blocking=False)`）。GPU 忙时立即返回 429，不再把请求线程挂在锁上。
- 前端增加 `generating` 标志，生成期间跳过轮询；轮询间隔从 2.5 秒放宽到 4 秒。
- 前端增加 `lastOk` 时间戳：连续 8 秒无响应才显示"后端无响应"，避免生成期间误报。

**验证：** 生成进行中（24.6 秒）并发 40 次、每秒 1 次的 health 轮询，**全部返回 200，零超时**。模型加载期间（`loading:"moss"`）health 同样秒回。双向切换 MOSS→Qwen 30 秒、Qwen→MOSS 21 秒，均正常。

**教训：** 在 ROCm/DTK 这类后端上，**永远不要从 HTTP 请求线程直接调 torch/CUDA API**。只让单一工作线程接触 GPU，其它线程读缓存。

---

## 8. 未做的事情（有意为之）

- **没装 `causal_conv1d`**：源码编译失败（需 CUDA/HIP 头文件匹配）。收益只在文本编码首次那 6.6 秒，风险是可能破坏现有环境。不装。
- **没装 `fla`**：当年 0.4.2 在 gfx936 上前向全零、反向 NaN。现在纯 torch 路径验证正常，没有理由引入已知会炸的依赖。
- **没做 fp32 残差流补丁**：默认 bf16 画质已可接受，且改动会偏离官方训练配置。
- **没跑 ComfyUI**：ComfyUI 有自己独立的 Qwen3.5 实现（`comfy/text_encoders/qwen35.py`，纯 PyTorch GDN），理论上是另一条可走的路，但对 UI 需求没必要。

---

## 9. 文件清单

本地：

| 路径 | 作用 |
|---|---|
| `tools/ui_server.py` | **统一 UI 后端**（双模型切换） |
| `tools/moss_infer.py` | 单次推理 + 计时 |
| `tools/moss_te_check.py` | 文本编码器数值验证（GDN 是否正常） |
| `tools/moss_timestep_check.py` | fp64 时间步验证 |
| `tools/moss_breakdown.py` | 耗时拆解（编码 / 去噪 / VAE） |
| `tools/anime_compare.py` | 双模型同提示词对比 |
| `tools/remote_tunnel.py` | 长驻会话 + 8081 端口转发 |

远端：

| 路径 | 作用 |
|---|---|
| `/workspace/models/MOSS-Image-SFT` | 26GB 权重 |
| `tools/ui_server.py` | UI 后端 |
| `/workspace/bench/gallery/` | 生成的图 + 元数据 |
| `/workspace/bench/moss/` | 测试图与脚本 |

---

## 10. 验收记录

| 项目 | 结果 |
|---|---|
| 文本编码器 GDN | 纯 torch 路径，`nan=False inf=False`，std≈2.0 |
| 跨 prompt 区分度 | 余弦 0.73 / 0.87 / 0.75（无坍缩） |
| fp64 时间步 | 每个时间步 L2=11.314（无归零） |
| 1024²/20步 g3.0 | 19.8 s，峰值 28.2 GiB |
| 1024²/20步 g1.0 | 9.6 s |
| 1328²/20步 | 37.6 s，峰值 31.3 GiB |
| 中文手写文字 | 三行便签逐字全对 |
| 二次元生成 | 通过，细节强于 Qwen |
| 模型切换 | MOSS→Qwen 20s，Qwen→MOSS 17s |
| 浏览器验证 | 通过（真实点击切换、参数联动、图库分标签） |
| 并发健康检查 | 生成中 40 次轮询全部 200，零超时 |
| 双向切换 | MOSS→Qwen 30s，Qwen→MOSS 21s |

---

## 12. 二次元提示词库

MOSS 对 Danbooru 风格的 tag 序列和自然语言都吃得很好。实测下来这几类效果最好：

### 12.1 核心质量 tag（几乎必加）

```
masterpiece, best quality, highly detailed, absurdres, newest
```

负面词（guidance>1 时生效）：

```
nsfw, worst quality, low quality, bad anatomy, bad hands, extra fingers, watermark, signature, jpeg artifacts
```

### 12.2 场景配方

**A. 和风意境**（实测效果最好的一类，光影层次强）

```
masterpiece, best quality, anime illustration of a girl in a white cheongsam with
golden embroidery, long black hair, red eyes, standing in a bamboo forest,
dappled sunlight through leaves, god rays, intricate hair ornament with tassel,
cel shading, detailed eyes, cinematic lighting
```

**B. 赛博霓虹**（MOSS 的强项，色彩饱和度高）

```
masterpiece, best quality, cyberpunk anime girl with neon blue twin tails,
holographic jacket, standing in a rainy neon-lit alley at night, reflections
on wet ground, glowing signs, bokeh, chromatic aberration, highly detailed
```

**C. 樱花少女**（最稳的萌系配方）

```
masterpiece, best quality, anime illustration of a young girl with long silver
hair and blue eyes, white sailor uniform, under cherry blossoms, petals falling,
soft sunlight, detailed eyes, cel shading, clean lineart, vibrant colors
```

**D. 战斗动态**（测结构能力）

```
masterpiece, best quality, dynamic action shot, anime girl with crimson hair
wielding a glowing katana, mid-slash, sparks and debris flying, dramatic
perspective, motion blur, intense expression, detailed armor, cinematic
```

**E. 日常治愈**（柔和光线，测色彩过渡）

```
masterpiece, best quality, cozy anime scene, girl with brown bob hair reading
a book by a window, afternoon sunlight, dust motes in the air, warm color
palette, soft shading, detailed interior, slice of life
```

**F. 中式古风**（MOSS 的中文语料优势）

```
masterpiece, best quality, 古风插画，一位身着红色广袖流仙裙的少女，长发及腰，
站在雪中红梅树下，水墨风格的远山，飘落的雪花，精致发簪，工笔画质感，细腻笔触
```

### 12.3 带文字的画面（MOSS 独门绝技）

MOSS 能把文字画对，这是 Qwen 做不到的。建议明确指定语言和内容：

```
anime illustration, a cheerful girl with pink twin tails holding a wooden sign
that reads "欢迎光临" in bold Chinese characters, colorful festival background
with lanterns and sparkles, cel shading, clean lineart, masterpiece
```

```
anime illustration, a bookstore scene, a girl holding a book titled "雪国" ,
warm lighting, detailed bookshelves, soft shading, masterpiece, best quality
```

### 12.4 参数建议

| 目标 | steps | guidance | 耗时（1024²） |
|---|---|---|---|
| 快速试稿找构图 | 20 | 1.0 | ~9.6 s |
| 常规出图（推荐） | 20 | 3.0 | ~19 s |
| 精致成品 | 30 | 3.0–4.0 | ~33 s |
| 极细节 / 大图 | 30 | 3.0 | 1328² 约 38 s |

- **guidance 1.0 会让色彩变淡**，二次元想浓郁就保持 3.0。
- **20 与 50 步在二次元上几乎无差别**，别按官方默认 50 步浪费时间。
- `slg_layers` 默认已用 22 层跳层，不用手动改。
- 中文提示词也能用，但混用中英时把质量 tag 放英文、描述放中文效果更好。
