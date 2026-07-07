# NewLoader 重构与移植指南

目标：把**旧 loader（`rtp_llm/model_loader/`，声明式 `ModelDeployWeightInfo`）能加载并推理的所有模型**，在**新 loader（`rtp_llm/models_py/`，vLLM 风格 `model.load_weights()`）**中重构完成，并逐一验证权重一致 + 推理等价。

---

## 0. 两套体系的本质差异

| | 旧 loader | 新 loader |
|---|---|---|
| 入口 | `rtp_llm/model_loader/loader.py` `ModelLoader` | `rtp_llm/models_py/model_loader.py` `NewModelLoader` |
| 权重描述 | 声明式：`ModelDeployWeightInfo` 子类列出每个 `WeightModule` 的源名→引擎名映射 | 命令式：每个 `nn.Module` 自己实现 `load_weights(weights_iter)` |
| 切分/量化 | loader 统一 `_split` / `_postprocess` | 各 layer（`layers/linear.py` 等）自带 TP 切分与量化融合 |
| 注册 | `rtp_llm/models/*` + `model_factory_register.py`（65 个 model_type） | `rtp_llm/models_py/__init__.py` `@register_model`（当前 7 个） |
| 启用 | 默认 | `USE_NEW_LOADER=1` 或 `model_config.use_new_loader=True`（见 `base_model.py:297`） |

**移植的基本单位是「架构族 / 权重类」而非 model_type**：一个架构跑通后，它的多个别名 model_type 往往只是 `__init__.py` 里加一行 `register_model("alias")(SameClass)`。

---

## 1. 当前覆盖与差距

新 loader 已注册（`rtp_llm/models_py/__init__.py`）：
`qwen_2, qwen_3, qwen_3_tool, qwen_3_moe, qwen3_coder_moe, deepseek_v32, qwen2_vl`

可复用的实现：
- `new_models/qwen3/`：稠密 attention + gate/up MLP + RMSNorm + tied embedding —— **稠密模型范本**
- `new_models/qwen3_moe/`：`BaseMoEExperts` + EP + FP8/FP4/W4A8 —— **MoE 范本**
- `new_models/deepseek_v3/`：MLA + 稀疏 MoE + 共享专家 —— **DeepSeek/MLA 范本**
- `new_models/qwen2_vl/`：vision + language 融合 —— **多模态范本**

旧 loader 共 65 个 model_type。运行 `check_imports.py` 可打印精确的「DONE / TODO」差距清单。

---

## 2. 可复用积木（先找现成的，不要重写）

`rtp_llm/models_py/layers/`：
- `linear.py`：`ColumnParallelLinear` / `RowParallelLinear` / `MergedColumnParallelLinear`(gate+up) / `QKVParallelLinear`，自带 TP 切分 + 量化
- `embedding.py`：`VocabParallelEmbedding` / `ParallelLMHead`
- `norm.py`：`RMSNorm` / `LayerNorm`（支持 gamma/beta 别名）
- `moe_experts.py`：`BaseMoEExperts`（8 种 FP8 量化族 + EP）

`rtp_llm/models_py/`：
- `module_base.py` `RtpModule`：`load_weights` 流式递归分派——叶子模块覆盖它即可
- `weight_mapper.py` `WeightsMapper`：HF 名 → 引擎名（prefix/exact/regex 映射）
- `model_desc/module_base.py` `GptModelBase`：FMHA / KV cache 支持的模型基类
- `quant_methods/`：`QuantizationConfig` + FP8 全家桶

> 记忆参考：优先从现成实现/上游移植，不要靠慢编译盲改。

---

## 3. 移植一个模型的标准任务清单

每个架构族一个任务，照抄 `new_models/qwen3/language.py` 的结构：

1. **建目录** `new_models/<arch>/{__init__.py, language.py}`（复杂 attention 可拆 `attention.py`）。
2. **对照旧权重类**：打开该 model_type 的 `get_weight_cls()`（旧 `ModelDeployWeightInfo` 子类），逐条把「源 ckpt 名 → 引擎名 → 切分规则 → 量化」翻译成新模块的拼装。这是移植的真正工作量来源，**别猜，照旧映射抄**。
3. **拼装子模块**：用 §2 的积木搭 `embed_tokens / layers[DecoderLayer] / norm / lm_head`。
4. **写 `load_weights`**：设 `WEIGHTS_MAPPER`（如 `prefix_mapping={"model.": ""}`），`super().load_weights(mapped_iter)` 走流式分派；tied embedding 等特例单独处理。
5. **`process_weights_after_loading`**（如需）：量化融合 / QKV 融合等冷路径后处理。
6. **注册**：`__init__.py` 里 `register_model("<model_type>")(XxxForCausalLM)`，所有别名 model_type 各加一行。
7. **写 CPU 单测**：仿 `new_models/qwen3_moe/test/test_qwen3_moe_load.py`，纯 CPU 验证切分/打包契约（Claude 这侧改不了运行，但单测代码要交付，由用户跑）。

---

## 4. 验证回路（每个模型都要走完）

Claude 在 Mac 只能读写代码，**编译/加载在远程容器跑**。验证分三档，由弱到强：

> ⚠️ 本仓库是 bazel 工程，python 脚本**必须 `bazel run`**（裸 `python3` 会缺编译出的
> `rtp_llm.ops`，报 `No module named 'rtp_llm'`）。只有纯 stdlib 的 `compare_dumps.py` 例外，
> 它在 Mac 侧直接 `python3` 跑。

**档 A — 静态/导入自检（容器）**
```bash
# (1) bazel run 不继承 .bazelrc 的 --action_env LD_LIBRARY_PATH，需手动导出，
#     否则 GPU torch 报 libcupti.so.12 not found：
export LD_LIBRARY_PATH="/lib64:/opt/conda310/lib/:/usr/local/cuda/compat/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs/:/usr/local/cuda/extras/CUPTI/lib64/:${LD_LIBRARY_PATH:-}"
# (2) 本容器 /lib 下的 libnvidia-ml.so.1 是 0 字节坏链接，preload /usr/lib64 下的真库，
#     否则 librtp_compute_ops 报 "libnvidia-ml.so.1: file too short"：
export LD_PRELOAD="$(ls -S /usr/lib64/libnvidia-ml.so.* | head -1):${LD_PRELOAD:-}"
# (3) 必须带匹配本容器的 arch config：x86 H20 -> cuda12_9 / arm -> cuda12_9_arm / AMD -> rocm
bazel run --config=cuda12_9 //rtp_llm/models_py/tools/newloader_port:check_imports
```
确认 main 合并后没破导入、新 model_type 已注册，并打印 DONE/TODO 差距清单。
`run_dump.sh`（档 B）已内置以上 (1)(2) 环境修复，无需手动 export。

**档 B — 权重指纹比对（容器跑 dump，Mac 比对）** ← 主力验证
```bash
# 容器内，仓库根目录：
bash rtp_llm/models_py/tools/newloader_port/run_dump.sh <model_type> <ckpt_path>
#   -> /tmp/newloader_cmp/<model_type>/{old,new}/rank*.json  （mutagen 同步回 Mac）
```
```bash
# Mac 侧（纯 stdlib，Claude 自己就能跑）：
python rtp_llm/models_py/tools/newloader_port/compare_dumps.py \
    --old-dir /tmp/newloader_cmp/<model_type>/old \
    --new-dir /tmp/newloader_cmp/<model_type>/new
```
比对逻辑：先按 **md5 内容指纹**做精确匹配，再对剩余张量按 **shape 分桶 + mean/std/absmax 容差**配对（处理 QKV 融合/拆分这类布局差异）。
- `VERDICT: PASS` 且无 `OLD-only` 张量 → 权重加载正确。
- 有 `OLD-only` 张量 → 那就是**漏加载/数值错**的权重，按名字回到旧权重类对照修。

**档 C — 推理等价（容器，金标准）**
同一 prompt 分别用旧/新 loader 跑一次 forward，比对 logits / 生成 token。档 B 过了但档 C 不过，通常是 forward 计算图（而非权重）差异。

---

## 5. 建议的分波路线（按「解锁 model_type 性价比」排序）

> 每波内：移植 → CPU 单测 → 档 B 比对 → 档 C 等价 → 注册别名。

- **Wave 0｜去风险**：跑 `check_imports.py`；对已完成的 7 个 model_type 跑档 B，确认 main 合并后基线仍 PASS。
- **Wave 1｜稠密大头**：LLaMA 族（`llama, mistral, aquila, baichuan(2), xverse, internlm(2), gemma, cohere, phi, yi…`）+ Qwen 稠密变体（`qwen, qwen_7b/13b/1b8, qwen_agent, qwen_tool, qwen35_dense…`）。多数复用 qwen3 范本，按族实现、按别名注册。
- **Wave 2｜稠密 GLM + 常规 MoE**：ChatGLM/GLM（`chat_glm_2/3, chatglm4, glm_5`）；MoE（`mixtral, qwen_2_moe, qwen35_moe, glm4_moe`）复用 `BaseMoEExperts`。
- **Wave 3｜DeepSeek/MLA 族**：`deepseek2/3, deepseek_v31, kimi_k2/k25` 复用 deepseek_v3 的 MLA+MoE。
- **Wave 4｜多模态**：`llava, qwen2_5_vl, qwen3_vl, qwen3_vl_moe, deepseek_vl_v2, chatglm4v` 复用 qwen2_vl 范本。
- **Wave 5｜长尾**：encoder/bert（`jina_bert_code, megatron_bert, qwen_2_embedding`）、audio（`qwen_v2_audio, cosyvoice_qwen`）、投机解码头（`*-mtp, qwen_3_moe_eagle3`）、线性注意力（`kimi_linear, qwen3_next`）。

实际优先级以「用户当前真正要拉起的模型」为准，可随时插队。

---

## 6. main 合并后的已知检查点

- `new_models/qwen3_moe/language.py:239` 从旧体系 import `quantize_weight_to_int4b`（文件存在，确认未改名）。
- `new_models/deepseek_v3/language.py:21`、多个 `model_desc/*.py` import `from rtp_llm.model_loader.model_weight_info import ModelWeights`（确认仍在）。
- 用 `check_imports.py` 一次性兜底所有注册模型的导入。
