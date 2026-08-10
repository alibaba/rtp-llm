# Kimi K3 运行参数

Kimi K3 只支持 PD 分离的 TP8/DP1/EP8 生产拓扑。Prefill 与 Decode
的 KDA、MLA、MoE 后端由模型根据 PD 角色选择，不通过环境变量切换。

`example/start_kimi_k3_pd.sh` 是受约束的启动入口。它要求 checkpoint
位于本地数据盘，并显式使用 `LOAD_METHOD=fastsafetensors`。

## 模型运行参数

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `KIMI_K3_FUSED_AG_GEMM` | `auto` | Prefill SP 的融合 AllGather + GEMM；可选 `auto`/`off`/`force` |
| `KIMI_K3_BATCHED_KDA_DECODE` | optimized Decode 为 `1` | 启用批量 KDA Decode |
| `KIMI_K3_MEGA_MAX_TOKENS_PER_RANK` | launcher 为 `8192` | DeepGEMM MegaMoE 每 rank token 容量 |
| `KIMI_K3_DEBUG` | `0` | 打开 K3 Decode SP 与 PD cache 传输诊断日志 |
| `KIMI_K3_TENSOR_DUMP` | 未设置 | 按 spec 导出精度定位张量；正常生产不设置 |
| `KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS` | `0` | 允许精度/性能工具使用固定 token IDs；正常生产不得开启 |

## 启动与容量参数

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `KIMI_K3_EXECUTION_MODE` | `optimized` | `optimized` 或 `accuracy`；后者默认关闭 batched KDA Decode |
| `KIMI_K3_MAX_SEQ_LEN` | `16384` | 服务最大序列长度 |
| `KIMI_K3_KV_CACHE_MEM_MB` | `8192` | KV/cache state 预留显存 |
| `KIMI_K3_REUSE_CACHE` | `0` | PD 两端一致开启 prefix cache 复用 |
| `KIMI_K3_DEEPGEMM_JIT_COMPILER` | `auto` | `auto`/`nvcc`/`nvrtc` |
| `KIMI_K3_RUN_ROOT` | `${TMPDIR}/kimi-k3-pd` | 日志和运行产物根目录 |
| `KIMI_K3_TMPDIR` | `/tmp/kimi-k3-pd-<role>` | 每角色的短路径运行目录 |
| `KIMI_K3_FLASHINFER_WORKSPACE_BASE` | `RUN_ROOT` 下 | 隔离 FlashInfer JIT cache |
| `KIMI_K3_FLASHINFER_CUDA_ARCH_LIST` | `10.3a` | B300/SM103a FlashInfer JIT 架构 |
| `KIMI_K3_SERVER_BINARY` | `bazel-bin/rtp_llm/rtp_llm_server` | 预构建服务入口 |
| `KIMI_K3_SKIP_BUILD` | `1` | `0` 仅允许在 L20-dev-115 的 `lhc_GPU` 内构建 |
| `KIMI_K3_BAZEL_OUTPUT_BASE` | 未设置 | 复用已有 CUDA13/SM10x Bazel output base |
| `KIMI_K3_DRY_RUN` | `0` | 只打印最终命令，不启动服务 |

## 固定生产选择

- Prefill KDA：cuLA。
- Decode KDA：Triton recurrent kernel。
- Prefill MLA：FlashMLA；Decode MLA：FlashInfer paged MLA。
- KDA 通信：`rs_ag`。
- MoE：DeepGEMM MegaMoE。
- 序列并行：TP8/EP8 生产路径恒开启。
- 性能融合 kernel：恒开启。

下列旧变量已无运行时读取，不应再出现在启动命令中：

```text
KIMI_K3_KDA_BACKEND
KIMI_K3_MLA_BACKEND
KIMI_K3_MOE_BACKEND
KIMI_K3_KDA_COMM_BACKEND
KIMI_K3_PERF_FUSIONS
KIMI_K3_SP_MOE
KIMI_K3_USE_HOST_METADATA
KIMI_K3_PERF_MODE
KIMI_K3_FASTSAFETENSORS_STREAMING
KIMI_K3_FASTSAFETENSORS_FILES_PER_BATCH
KIMI_K3_PD_TRACE_LOG_ENABLE
KIMI_K3_DECODE_TOPOLOGY
```

64K timeline 的固定后端会写入 manifest，但不再伪装成可切换参数。
每个测量组先执行一次物化请求、至少十次完整 64K 预热，再单独
执行 profiler 预热与正式 trace。
