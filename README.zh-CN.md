# 在海光 DCU（gfx936）上运行 MOSS-Image

**首份公开运行报告、实测性能数据，以及一套部署工具。**

[English](README.md)

---

## 这件事是什么，不是什么

**不是移植。** MOSS-Image 在海光 DCU 上跑通，**模型代码零改动**：没有改 kernel、
没有 workaround、没有 fork 上游代码。上游的 `pipeline_moss_image.py` 和上游
`transformers` 的 Qwen3.5 实现，原样可用。

实际贡献是另外四件事：

1. **首次跑通记录。** 截稿时，公开渠道没有任何 MOSS-Image 在 ROCm / AMD / 海光
   上运行的报告。在上游 issue 区和 Hugging Face discussions 搜索 `rocm`、`amd`、
   `dcu`、`hygon` 均为零结果。本仓库记录了一套可用配置。
2. **排查掉两个隐性地雷。** 见[数值风险](#数值风险)。这两个问题的共同特征是
   **出错但不报错**，属于最难排查的一类。
3. **在服务层发现一个真实 bug。** 从 HTTP 请求线程并发访问 HIP runtime 会死锁。
   任何在这些驱动上用 Web 界面跑扩散模型的人都会踩。见[死锁](#rocmpdtk-上的死锁)。
4. **在真实硬件上的实测数据与调优参数**，后来者不必重新推导。

如果你只想知道"怎么让 MOSS-Image 在 DCU 上跑起来"，答案是：装厂商版 PyTorch，
装 `diffusers` 和 `transformers`，直接跑。本仓库的价值在于让你**一开始就知道**
这个答案，以及绕开那些坑。

---

## 实测环境

| 项目 | 值 |
|---|---|
| 加速卡 | 1× 海光 HCU（"BW"），80 CU，gfx936 |
| 显存 | 65520 MiB（64.0 GiB） |
| 驱动 | 6.3.31-V1.5.3.beta |
| 工具链 | DTK 26.04 |
| PyTorch | `2.7.1+das.opt1.dtk2604`（厂商版，HIP 6.3.26093） |
| diffusers | 0.43.0 |
| transformers | 5.23.0 |
| 主机限制 | 59 GiB 内存、15 核（cgroup）、1.8 TB 磁盘 |

### 实测算力（非规格表）

| 项目 | 结果 | A100 80GB 参考 |
|---|---|---|
| bf16 matmul 8192³ | **307.95 TFLOP/s** | 312 TFLOP/s（稠密） |
| fp16 matmul 8192³ | 307.70 TFLOP/s | 312 TFLOP/s |
| fp32 matmul 4096³ | 54.48 TFLOP/s | 19.5（无 TF32） |
| 显存拷贝带宽 | **1353.6 GB/s** | 2039 GB/s |
| bf16 SDPA，seq 4096 | 172.33 TFLOP/s | — |

bf16 算力是 A100 级，带宽约为三分之二。扩散模型算力密集、对带宽相对不敏感，
所以这张卡在这里表现很好。复现：`tools/bench_bw.py`。

---

## 数值风险

以下两个问题都会**静默产出错误图像**而不报错。均已实测，
见 `tools/moss_te_check.py` 与 `tools/moss_timestep_check.py`。

### 1. 缺少优化 kernel 时的 Qwen3.5 GDN 线性注意力

文本编码器是 Qwen3.5-9B，其结构为每 4 层中有 3 层是 Gated DeltaNet（线性注意力）。
快速路径需要 `fla`（flash-linear-attention）和 `causal_conv1d`。

两者**都不是硬依赖**。`transformers` 使用
`use_kernel_func_from_hub_with_fallback`，包缺失时自动降级到参考 PyTorch 实现，
并打印：

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch
implementation because `flash-linear-attention` is not installed.
This is correct but much slower; install `flash-linear-attention` for the
optimized kernel.
```

在这张卡上实测纯 PyTorch 路径：

- 无 NaN、无 Inf、无零值坍缩
- 输出 std ≈ 2.0，逐位置 std 0.52–29.1
- 跨 prompt 余弦相似度 0.73 / 0.87 / 0.75（无嵌入坍缩）
- 首次文本编码 **6.64 秒**，第二次 **0.10 秒**

代价是延迟，不是正确性。这一点很重要：同一个项目早期在这张卡上遇到过某个
`fla` 版本前向返回全零、反向 NaN —— 所以"能 import"不等于"算得对"。
`tools/` 里的探针才是判断正确性的依据。

### 2. float64 时间步嵌入

上游 pipeline 用 float64 计算时间步正弦嵌入：

```python
freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
phase = times.double()[:, None] * freq
```

在不支持 float64 的后端上，这可能**静默返回全零**，导致所有时间步相同、
输出纯噪声且不抛异常。MPS 和部分 NPU/XPU 栈上已有此问题的公开记录。

在这张海光卡上 float64 工作正常：

```
input times: [1000.0, 976.0, 500.0, 24.0, 0.0]
per-timestep L2 norm: [11.314, 11.314, 11.314, 11.314, 11.314]
```

如果要换到别的加速卡，**先跑 `tools/moss_timestep_check.py`** 再相信任何输出。
范数为零意味着你生成的每一张图都是噪声。

---

## 性能

MOSS-Image，1024×1024，`uncond_skip_layers=[22]`，模型常驻显存。

| 步数 | guidance | 耗时 | 每步 | 峰值显存 |
|---|---|---|---|---|
| 20 | 3.0 | **19.8 s** | 0.66 s | 28.2 GiB |
| 50 | 3.0 | 45.9 s | 0.79 s | 28.2 GiB |
| 20 | 1.0 | **9.6 s** | 0.48 s | 28.2 GiB |

其他分辨率（20 步，guidance 3.0）：

| 分辨率 | 耗时 | 峰值显存 |
|---|---|---|
| 1328×1328 | 37.6 s | 31.3 GiB |
| 1536×1024 | 33.3 s | 30.2 GiB |

模型常驻占用 25.4 GiB；加载耗时 8–20 秒（取决于页缓存）。

### 为什么比"10GB 的 transformer"慢

"10GB" 只是 DiT。完整管线 26 GB：

| 组件 | 磁盘 | 说明 |
|---|---|---|
| transformer（DiT，42 层，dim 3072） | 9.5 GB | 常说的 10GB |
| text_encoder（Qwen3.5-9B，8.39B 参数） | 16 GB | 与 DiT 同量级 |
| vae（FLUX.2 small） | 238 MB | 设计上强制 fp32 |

慢有两个独立原因：

1. **CFG 使计算翻倍。** `guidance_scale != 1` 时每步跑两次 DiT 前向（条件与无条件）。
   把 guidance 设为 1.0 可省一半时间：20 步从 19.8 秒降到 9.6 秒。
2. **首次文本编码 6.6 秒**，因为 GDN 走的是降级路径。

作为参照，Qwen-Image 2.1 在 1024² 下 `cfg=1.0` 是每步 0.40 秒，因为它每步只跑一次前向。

### 调优参数

| 目标 | 步数 | guidance | 耗时 |
|---|---|---|---|
| 快速试稿 | 20 | 1.0 | ~9.6 s |
| 常规出图（推荐） | 20 | 3.0 | ~19 s |
| 高细节 | 30 | 3.0–4.0 | ~33 s |

上游推荐 ≥50 步。实测二次元题材下 20 步与 50 步肉眼无差别，默认 50 步通常是浪费时间。
guidance > 1 色彩更浓；1.0 明显偏灰。

`uncond_skip_layers` **不是**禁用负向提示词。负向词仍会被编码并作用于无条件分支，
跳层只是让该分支跳过第 22 层（及其他指定层）以节省算力。两者可以同时使用。

---

## 快速开始

必须先装厂商版 PyTorch。从 PyPI 安装通用 torch 会覆盖 DTK 构建并破坏加速卡路径。

```bash
# 1. 环境（复用镜像里已有的厂商版 torch）
python3 -m venv --system-site-packages /workspace/venv-moss
/workspace/venv-moss/bin/pip install -r requirements.txt

# 2. 权重（26 GB）
/workspace/venv-moss/bin/modelscope download \
  --model openmoss/MOSS-Image-SFT \
  --local_dir /workspace/models/MOSS-Image-SFT

# 3. 先做数值自检 —— 不要跳过
/workspace/venv-moss/bin/python tools/moss_te_check.py
/workspace/venv-moss/bin/python tools/moss_timestep_check.py

# 4. 生成
/workspace/venv-moss/bin/python tools/moss_infer.py \
  --prompt "masterpiece, anime illustration of a girl under cherry blossoms" \
  --steps 20 --guidance 3.0 --out /workspace/bench/out.png
```

最小推理代码：

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/workspace/models/MOSS-Image-SFT",
    custom_pipeline="pipeline_moss_image",   # 该文件在模型仓库里，不在 diffusers 中
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

几个花了时间才发现的细节：

- `custom_pipeline="pipeline_moss_image"` 是必须的，pipeline 类随模型仓库分发。
- `torch_dtype` 必须传字典。VAE 设计上就是 fp32（`force_upcast: true`）。
- tokenizer 在 `tokenizer/` 目录，不在 `text_encoder/` 下。
- 宽高必须是 16 的倍数。

---

## ROCm/DTK 上的死锁

如果你打算在这类卡上用 Web 界面服务任何扩散模型，这段值得读。

**现象。** Web UI 失去响应。进程还在，`hy-smi` 显示显存仍被占用，但端口没有监听，
在远端机器上 `curl 127.0.0.1` 同样失败。

**原因。** 两个错误叠加：

1. 健康检查接口每次请求都调用 `torch.cuda.memory_allocated()`。前端会轮询它。
   每次轮询都在一个 HTTP 处理线程里进入 HIP runtime。当另一个线程正在跑生成时，
   并发进入 HIP runtime 会死锁。
2. 生成任务持有全局锁长达 25–50 秒，轮询请求排队等锁，线程不断堆积，
   最终 socket 不再接受新连接。

**修复。** GPU 调用只留在一个工作线程，其余一律读缓存：

- 健康检查返回纯字典拷贝，绝不接触 torch
- 显存数据只在持有 GPU 锁时采样进该字典
- GPU 锁采用**非阻塞**获取，忙时立即返回 HTTP 429，而不是把请求线程挂在 torch 上
- 前端在生成期间跳过轮询

**验证。** 在一次 24.6 秒的生成过程中，以每秒 1 次并发 40 次健康检查，
**全部返回 200，零超时**。模型加载期间健康检查同样秒回。

**规则。** 在 ROCm/DTK 上，不要从请求线程调用 torch 或 CUDA/HIP API。
单一 GPU 所有者线程，其它线程读缓存。

---

## Web 界面

`tools/ui_server.py` 提供单页界面，接入两个后端并可切换——因为 Qwen-Image 2.1
（30.2 GiB）与 MOSS-Image（25.4 GiB）在 64 GB 卡上无法同时常驻。

```bash
# 在远端机器上，放入长驻会话（SSH 断开后后台进程会被平台清理，nohup 不够）
python3 tools/ui_server.py

# 在本机转发端口
python3 tools/remote_tunnel.py 'cd /workspace && exec python3 ui_server.py'
# 然后打开 http://127.0.0.1:8081
```

配置全部来自环境变量，见 `.env.example`。仓库里不含任何特定机器的信息。

`tools/remote_exec.py`、`remote_run.py`、`remote_tunnel.py`、`remote_put.py`
读取 `MOSS_HOST` / `MOSS_PORT` / `MOSS_USER` / `MOSS_PASS_FILE`。
不设置 `MOSS_PASS_FILE` 即使用密钥或 agent 认证，这是推荐做法。

---

## 画质说明

- **中文手写文字是真优势。** 上游那个冰箱便签提示词（三行中文手写）逐字渲染正确，
  且是自然的书写笔触。
- **二次元表现强。** 眼睛层次、发丝、赛璐璐上色。与 Qwen-Image 2.1 用同一提示词、
  同一种子对比，MOSS 出的是更紧的近景特写、表面细节更多；Qwen 是更柔和的半身中景。
- **bf16 残差流。** 上游全程 bf16，issue #31 实测其预测误差比 fp32 残差流高
  2.4–3.5 倍。上游为与训练保持一致而拒绝修改。如果觉得画质偏软，把非
  `layers.*.proj.*` 的参数转 fp32 是已知的缓解手段，代价约 1.5% 速度。本项目未采用。

测试过的二次元提示词配方见 [docs/PROMPTS.md](docs/PROMPTS.md)。

---

## 有意未做的事

- **不装 `causal_conv1d`。** 针对此工具链从源码编译失败，收益仅是那 6.6 秒的首次编码，
  不值得冒破坏环境的风险。
- **不装 `fla`。** 该库早期版本在这张卡上前向返回全零、反向 NaN。
  纯 PyTorch 路径已验证正确，没有理由重新引入一个已知会失败的依赖。
- **不打 fp32 残差流补丁。** 默认 bf16 输出可接受，且与上游训练配置一致。
- **不跑 ComfyUI。** ComfyUI 自带纯 PyTorch 的 Qwen3.5 实现（含 GDN，
  见 `comfy/text_encoders/qwen35.py`），是一条可行替代路径，但本场景不需要。

---

## 目录结构

```
tools/
  remote_config.py         从环境变量读取 SSH 配置
  remote_exec.py           一次性远端命令
  remote_run.py            长驻远端会话（用于长任务）
  remote_tunnel.py         长驻会话 + 本地端口转发
  remote_put.py           通过 SSH 上传文件
  ui_server.py             Web 界面，双后端，可切换模型
  moss_infer.py            单次生成并计时
  moss_te_check.py         文本编码器数值验证
  moss_timestep_check.py   float64 时间步验证
  moss_breakdown.py        耗时拆解：编码 / 去噪 / VAE
  bench_bw.py              算力与带宽微基准
docs/
  DEPLOYMENT.md            完整部署记录与故障手册
  PROMPTS.md               实测二次元提示词配方
```

## 许可

本仓库为 MIT（见 `LICENSE`），不含任何模型权重。

- MOSS-Image：Apache 2.0，由 OpenMOSS 发布。
- Qwen-Image 2.1：Qwen Research License Agreement，**不是**开源许可。
  仅作为可选的对比后端引用。

## 致谢

使这件事成为可能的上游工作：[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)、
[Hugging Face diffusers](https://github.com/huggingface/diffusers)、
[transformers](https://github.com/huggingface/transformers) ——
尤其是它优雅的 `fla` 降级机制，没有它就无法做到零改动运行。

欢迎指正，特别是在其他 DCU 型号、ROCm 显卡上运行的结果，
或成功编译 `fla`/`causal_conv1d` 的经验。
