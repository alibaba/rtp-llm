# Kimi K3 环境变量清单

精简前代码里读取 **35 个** `KIMI_K3_*`,现在是 **17 个**。这份清单列出每一个:
谁在读、默认值、为什么保留。

一次运行的完整配置请用 `example/kimi_k3_pd_launch.sh`,它把每个值显式写出来 ——
`start_kimi_k3_pd.sh` 会从 `EXECUTION_MODE` × 拓扑 × 角色推导十几个默认值,
这些值不出现在命令行上,是误判的常见来源。

---

## 容量与部署(4)

| 变量 | 默认 | 读取处 | 说明 |
|---|---|---|---|
| `KIMI_K3_MEGA_MAX_TOKENS_PER_RANK` | `65536` | `model_desc/kimi_k3.py` | DeepGEMM MegaMoE 对称环 buffer 的每 rank token 上限。**与实际 token 数不匹配会 illegal memory access,不是 OOM** —— 症状会把排查带偏 |
| `KIMI_K3_DEEPGEMM_EXPECTED_PATH` | — | `model_desc/kimi_k3.py` | 算子 overlay 路径校验 |
| `KIMI_K3_FASTSAFETENSORS_STREAMING` | `0` | `utils/database.py` | 逐批关闭 staging,控制加载期显存峰值 |
| `KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH` | — | `utils/database.py` | 每批文件数 |

## 实现选择(4)

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `KIMI_K3_KDA_BACKEND` | `kernel` \| `cula` | `kernel` | **Prefill 必须 `cula`**。`kernel` × `PERF_FUSIONS=1` 实测 0/4(见下) |
| `KIMI_K3_MLA_BACKEND` | `kernel` \| `flashmla` | `kernel` | `kernel` = FlashInfer paged;`flashmla` 仅 Prefill 可用 |
| `KIMI_K3_MOE_BACKEND` | `deepep` \| `deep_gemm_mega` | `deepep` | 生产用 `deep_gemm_mega`;`deepep` 在 93 层 Decode 会 OOM |
| `KIMI_K3_KDA_COMM_BACKEND` | `rs_ag` | `rs_ag` | 代码显式拒绝其它值 |

## 性能开关(4)

| 变量 | 默认 | 说明 |
|---|---|---|
| `KIMI_K3_PERF_FUSIONS` | `0` | Triton 融合 kernel。**与 `KDA_BACKEND` 强耦合,只在 `cula` 下安全** |
| `KIMI_K3_SP_MOE` | `0` | token 序列并行 MoE。Decode TP8/EP8 硬要求为 `1`,否则 launcher 直接 die |
| `KIMI_K3_BATCHED_KDA_DECODE` | `0` | 批量 KDA decode。实测与 fusions 的精度问题**无关** |
| `KIMI_K3_USE_HOST_METADATA` | `0` | host 侧 metadata 避免 device 同步。实测**对精度无影响**(档 A→B) |

## 诊断(4)

| 变量 | 说明 |
|---|---|
| `KIMI_K3_TENSOR_DUMP` | 逐算子张量 dump,单变量带完整 spec:<br>`<dir>[,rank=N][,mode=boundary\|semantic_full][,forward=N][,router=full][,token=N][,enable_file=PATH][,shard_bytes=N]`<br>替代原来 8 个 `ACCURACY_TRACE_*` |
| `KIMI_K3_DEBUG` | `1` 打开全部 K3 诊断日志。替代 `DECODE_SP_DEBUG` + `PD_TRACE_LOG_ENABLE` |
| `KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS` | 允许固定 token id 输入(精度套件必开),不开会被拒 |
| `KIMI_K3_PERF_MODE` | 严格性能路径校验,产品默认 `0`。会禁止某些合法组合 |

## 开发工具(1)

| 变量 | 说明 |
|---|---|
| `KIMI_K3_REQUIRE_DEEP_EP` | 只在 `example/kimi_k3_prefill_perf/server_main.py` |

---

## 已删除的 18 个

```
CPU_OFFLOAD_EXPERT_LAYER_START · DECODE_CPU_OFFLOAD_START · SP_LAST_HIDDEN_ONLY
KDA_CHUNK_STATE_BACKEND · KDA_A2A_SAFETY_GIB · KDA_FLA37_PRECOMPILED_DIR
ACCURACY_MODE(含 canonical / native_mla)
ACCURACY_CANONICAL_TP / _EP / _MLA · ACCURACY_LOCAL_EAGER_MLA
ACCURACY_RETAIN_FULL_TP_WEIGHTS
ACCURACY_TRACE_{DIR,RANK,MODE,FORWARD_INDEX,ENABLE_FILE,FULL_ROUTER,
                INPUT_TOKEN_ID,MAX_SHARD_BYTES}        -> TENSOR_DUMP
DECODE_SP_DEBUG + PD_TRACE_LOG_ENABLE                  -> DEBUG
KDA_BACKEND 的 reference / flash_kda / fla37_precompiled 三个取值
```

净删约 1900 行。

---

## 精度依据

93 层、官方权重 `9f62e4e9`、对封版 golden 全 16 条 × 3 次,115 Prefill + 114 Decode:

| 档 | KDA(P) | MLA(P) | FUSIONS | BATCHED | 结果 |
|---|---|---|---|---|---|
| A | kernel | kernel | 0 | 0 | 4/4 exact |
| B | kernel | kernel | 0 | 0 | 4/4 exact |
| **C** | kernel | kernel | **1** | 0 | **0/4** |
| **D** | kernel | kernel | 1 | **1** | **0/4**,边距与 C 逐条相同 |
| **E** | kernel | **flashmla** | 1 | 1 | **0/4** |
| F | **cula** | flashmla | 1 | 1 | 4/4 exact |
| G | **cula** | flashmla | **0** | 1 | 4/4 exact |
| H | **cula** | **kernel** | 1 | 1 | 4/4 exact ← 生产配置 |

C/D 边距逐条相同 → `BATCHED_KDA_DECODE` 与该 bug 无关;E 换 MLA 不救;F/G/H 换 cula 全对。
**结论:坏的是 `KDA_BACKEND=kernel` × `PERF_FUSIONS=1` 的组合。**

D 档 4 条全部在**第 1 个输出 token** 就分叉,边距 54/10/15/37 ulp,三次重跑一致 ——
量级远超舍入噪声。

## 尚未验证

- **CUDA Graph**:上述全部结果都在 `ENABLE_CUDA_GRAPH=0` 下取得,开启后的精度影响无数据
- **长生成的非确定性**:基线自身在 s01(140 token)/ s03(48)/ s08(241)上三次重跑结果不同,
  与本次改动无关,单独跟进
