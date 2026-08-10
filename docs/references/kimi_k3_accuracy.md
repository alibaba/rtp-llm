# Kimi K3 四层模型逐算子精度方案

本文定义 Kimi K3 在 RTP-LLM 中的精度基线、逐算子采样边界、验收标准和优化回归顺序。目标不是只比较最终 token，而是从输入开始找到第一个发生偏差的语义 tensor，并区分模型公式错误、分布式布局错误、状态传递错误和优化 kernel 误差。

## 1. 被测对象

- RTP-LLM: 当前仓库的 `rtp_llm/models_py/model_desc/kimi_k3.py` 及 `rtp_llm/models_py/modules/kimi_k3/`。
- 四层 checkpoint: `/data5/kimi-k3-4layers`。
- PyTorch oracle: checkpoint 自带的 `modeling_kimi.py`，使用原始 `KimiRMSNorm`、`KimiDeltaAttention`、`KimiMLAAttention`、`KimiMoEGate`、`KimiSparseMoeBlock` 和 `KimiDecoderLayer`。
- FLA oracle: `fla-core 0.5.1` 的 `ShortConvolution`、`chunk_kda`、`fused_recurrent_kda` 和 `FusedRMSNormGated`。

四层结构是：

| Layer | Attention | MLP |
|---|---|---|
| 0 | KDA | Dense SiTU |
| 1 | KDA | Latent MoE, 896 experts, top-16 |
| 2 | KDA | Latent MoE, 896 experts, top-16 |
| 3 | Gated MLA | Latent MoE, 896 experts, top-16 |

关键配置为 hidden size 7168、96 个 attention heads、head dim 128。专家权重使用 MXFP4 E2M1，scale 使用 UE8M0。

## 2. Oracle 分层

精度门禁按四层推进。前一层不通过时，不允许用后一层的最终 token 结果掩盖问题。

1. Checkpoint contract
   - 校验 config、index/shard key-set、dtype 分布、层类型以及全部 896 个专家的 w1/w2/w3 packed/scale。
   - 校验官方 PyTorch 和 RTP 实现都存在对应算子及 trace 边界。
2. Formula oracle
   - 在 CPU 上用真实 checkpoint tensor 检查 RMSNorm、RMSGated、AttnRes、Router top-16 和 MXFP4 解码。
   - 用确定性小 shape 检查 SiTU、short conv、KDA prefill/decode state 和 MLA cache 连续性。
3. Canonical integration
   - 先跑 TP1/EP1，排除 collective 和 shard layout。
   - 再跑 TP8/EP8，并打开 canonical TP/EP/MLA，验证权重切分、all-gather/all-reduce、router layout 和 cache handoff。
4. Optimized backend A/B
   - 每次只打开一个变量，并与同一次输入的 canonical trace 比较。
   - 顺序为 KDA kernel -> FlashKDA/FLA recurrent -> MLA kernel -> DeepEP -> MegaMoE -> Sequence Parallel/perf fusion -> CUDA Graph capture/replay。

## 3. 固定输入和运行矩阵

输入 token 固定为：

```python
ids = [100 + ((index * 7919 + 17) % 160000) for index in range(length)]
```

固定 `do_sample=False`、`ignore_eos=True`、`random_seed=20260722`。每个 case 生成 4 个 token，因此包含一次 prefill 和三次 decode。每个 case 至少重复两次。

| Case | Trace mode | 用途 |
|---|---|---|
| 64 tokens | `semantic_full` | 快速逐 tensor 定位，包含完整 MLA attention matrix |
| 256 tokens | `semantic_full` | KDA chunk 边界和较大 MLA shape |
| 8192 tokens | `boundary` | TP8/EP8 集成门禁，控制 trace 体积 |
| 8192 tokens | `semantic_full` | TP1 长上下文状态与 cache 门禁 |

RTP trace 使用：

```bash
export KIMI_K3_TENSOR_DUMP=/path/to/trace,mode=semantic_full,rank=0,router=full
export KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS=1
```

Canonical TP/EP/MLA 对照模式已从代码里删除：它们各自额外引入一条分支，而每条
分支都会扩大"精度对不上"的可能原因集合。逐算子对照现在只靠上面的 trace，观测
的就是生产路径本身。

## 4. 逐算子边界

| 算子族 | 必须比较的 tensor |
|---|---|
| Embedding | `embedding` |
| RMSNorm | `attention_input`, `normalized_mlp_input`, `final_hidden` |
| Attention residual | `mlp_input`, layer `output`, `block_residual`, `output_attn_res` |
| KDA projection | q/k/v projected、raw gate、raw beta、output gate |
| Short conv | q/k/v conv output、prefill final state、decode cache input |
| KDA scan | prepared q/k、alpha、beta、core output、FP32 recurrent state |
| KDA output | gated RMSNorm output、o_proj output |
| MLA | query latent/query、compressed current、cache、scores/probabilities/context、output gate、output |
| Dense MLP | gate/up projection、SiTU、down projection |
| Router | sigmoid score、correction margin、top-16 IDs、routing weights、expert counts |
| Routed experts | routed input、expert sum、latent norm、latent projection output、routed output |
| Shared expert | gate/up、SiTU、down projection |
| Final | final hidden、logits、greedy token |

Router 不能只做 allclose。每个 token 都必须检查 top-16 集合、顺序、重复 ID、越界 ID 和归一化权重。Decode 必须额外比较 KDA conv/recurrent cache input 和 MLA cache，以便把“当前 token 算错”和“上一步状态已污染”分开。

## 5. 验收阈值

- shape 和 dtype 必须完全一致。
- input IDs、cu_seqlens、router IDs/counts 必须完全一致。
- FP32 control tensor 默认 `atol=1e-5, rtol=1e-4`。
- KDA chunk/recurrent 因归约顺序变化产生的 state 误差按模型 state 处理，使用 `atol=2e-2, rtol=2e-2`，但同时记录最大绝对误差。
- BF16 tensor 使用 `atol=2e-2, rtol=2e-2`，cosine 不低于 0.999。
- `final_hidden` 额外要求 NRMSE 不高于 0.01。
- logits 要求 max abs 不高于 0.08、cosine 不低于 0.999、NRMSE 不高于 0.01，并要求 greedy token 完全一致。
- 报告必须给出第一个失败 tensor；只给最终 token 结果不算逐算子验收完成。

## 6. 新鲜运行与证据隔离

每次验证必须在当前 worktree 上重新生成官方 golden 和 RTP trace。旧报告、旧
golden、其他 worktree 以及其他用户目录下的任何产物都不得作为 PASS 证据，也不得
复制进本轮运行目录。运行根目录必须记录当前 HEAD、dirty diff、官方源码/checkpoint
哈希、容器镜像、GPU 信息、完整命令和环境变量，并为每个 tensor artifact 保存校验和。

当前 HEAD 的门禁按以下顺序执行，每一步都保存独立 trace 和汇总 JSON：

1. 当前 HEAD canonical TP8/EP8，64 semantic-full 和 8192 boundary。
2. KDA `kernel` prefill 与 FLA recurrent decode。
3. MLA kernel，先关闭再打开 local eager diagnostic。
4. DeepEP routed expert，然后 MegaMoE。
5. Sequence Parallel 和 perf fusion。
6. CUDA Graph 第一次 capture、连续两次 replay，并检查 cache slot/state 不跨请求污染。

任何一步失败，都从报告的 `first_divergence` 开始修复并重跑同一层；不要直接跳到最终 logits 调阈值。

## 7. 本地只读命令

```bash
PYTHONNOUSERSITE=1 /opt/conda310/bin/python3 \
  tools/kimi_k3_accuracy/audit_checkpoint.py \
  --checkpoint /data5/kimi-k3-4layers \
  --hf-modeling /data5/kimi-k3-4layers/modeling_kimi.py \
  --rtp-modeling rtp_llm/models_py/model_desc/kimi_k3.py

PYTHONNOUSERSITE=1 /opt/conda310/bin/python3 \
  tools/kimi_k3_accuracy/formula_parity.py \
  --checkpoint /data5/kimi-k3-4layers \
  --repo-root .

PYTHONNOUSERSITE=1 /opt/conda310/bin/python3 \
  tools/kimi_k3_accuracy/compare_artifacts.py \
  /path/to/golden/prefill.safetensors \
  /path/to/rtp/prefill.safetensors \
  --repo-root . \
  --output /path/to/reports/prefill.json

PYTHONNOUSERSITE=1 /opt/conda310/bin/python3 \
  tools/kimi_k3_accuracy/summarize_reports.py \
  /path/to/reports/prefill.json /path/to/reports/decode-*.json \
  --output /path/to/operator_summary.json
```
