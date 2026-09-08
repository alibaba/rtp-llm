# cuda graph 与 offload 兼容性

日期：2026-09-08

## 结论

本任务停在阶段 2，没有进入双模式实现。

原因不是 graph 收益不足，而是当前带独立 `dsa_indexer_k` pool 的 DeepSeek-V3.2 runtime 无法完成 CUDA graph capture。A1 和 C 在 `enable_cuda_graph=1` 下都连续两次于 D 启动阶段失败，8 个 rank 报同一个错误：

```text
RuntimeError: independent indexer pool cannot derive its slot mapping
```

服务没有开放 HTTP 端口，graph-on 条件下没有发出任何请求。因此：

- A1 graph-off 基线已测得；
- A1 graph-on TPOT 不存在，graph 税无法计算；
- C graph-on 的纯短和 managed/mixed 路径都无法到达请求面；
- 没有 graph-on 输出 hash，不能做运行时正确性对照；
- `graph 收益 ≥2ms/token 或 ≥2%` 的阶段 3 门槛无法成立；
- 没有实现双模式，也没有为本任务修改引擎、`v32_ctx.cu`、Python hook 或 attention kernel。

当前失败是显式 capture 崩溃，不是静默错算。但静态路径显示，即使先修复 capture，C managed replay 仍有绕过实时 Python remap/fetch 的静默错算风险，不能直接打开 graph 上线。

## 实验配置

复用 fast path 实验：

| 项目 | 配置 |
|---|---|
| P | `11.17.131.204`，TP2×DP4，EP8 |
| D | `33.240.36.239`，TP1×DP8，EP8，12GiB KV/卡 |
| 路由 | 全部通过 `role_addrs` 固定到 D rank0 |
| 短请求 | 2,220 输入，1,024 输出，8 条，两波提交 |
| 长请求 | 62,830 输入，8,192 输出 |
| 公共参数 | `reuse_cache=0`、`max_seq_len=73728` |
| TPOT | `(decode_service_us - decode_first_token_us)/(output_len-1)` |

A1 与 C 都保留独立 `dsa_indexer_k` pool；唯一切换项是 `enable_cuda_graph=0/1`。

## 阶段 1：A1 graph 税

### Graph off 基线

| 条件 | Round 1 mean | Round 2 mean | 合并 mean / p50 / p95 |
|---|---:|---:|---:|
| A1 纯短 | 117.205 | 116.861 | **117.033 / 116.882 / 117.697** |
| A1 混批短请求 | 117.258 | 115.459 | **116.359 / 116.311 / 117.473** |

结果完整性：

- 纯短 16/16 成功；
- 混批短请求 16/16 成功；
- 两条长请求均完整输出 8,192 tokens；
- 短请求 hash 均为 `47229f5fe1b12614`；
- 长请求 hash 均为 `27887e7c2b6b7b51`；
- 时间窗内无 allocator、CUDA、RDMA 或服务错误。

### Graph on

A1 graph-on 冷启两次。两次均在模型加载后进入 capture，随后 8 个 D rank 全部退出：

```text
RuntimeError: independent indexer pool cannot derive its slot mapping

At:
  rtp_llm/models_py/modules/hybrid/indexer.py:_indexer_slot_mapping
  rtp_llm/models_py/modules/hybrid/indexer.py:_quantize_q_k
  rtp_llm/models_py/modules/hybrid/indexer.py:forward
  rtp_llm/models_py/modules/hybrid/mla_attention.py:_run_sparse_indexer
```

两次均为：

```text
HTTP ready: 0/8
requests sent: 0
```

因此 graph 收益不能用 `graph-off - graph-on` 计算。A1 本身当前就无法启用 graph，不能把该问题归因于 C offload hook。

## 阶段 2：C 兼容性

C graph-on 同样冷启两次，结果与 A1 一致：capture 阶段所有 D rank 报 `independent indexer pool cannot derive its slot mapping` 并退出，HTTP 0/8，未发请求。

兼容性矩阵：

| 模式 | Graph off | Graph on | 正确性结论 |
|---|---|---|---|
| A1 纯短 | 通过，16/16 | 启动 capture 失败 | graph-on 无输出可比 |
| A1 混批 | 通过，16/16 短 + 2/2 长 | 启动 capture 失败 | graph-on 无输出可比 |
| C `native_only` 纯短 | 既有 fast-path 结果通过：115.538ms | 启动 capture 失败 | 未到请求面，不存在自动 eager fallback 证据 |
| C managed/mixed | 既有结果通过：116.366ms | 启动 capture 失败 | 未到请求面，尚未发生静默错算，但也未验证 replay |

C 的实际失败模式是“capture fail-fast + 全 rank 退出”，不是自动 eager fallback，也不是请求返回错误文本。

## 根因

`Indexer._indexer_slot_mapping()` 优先读取 `fmha_params.indexer_slot_mapping`。capture 输入中该映射不存在时，它尝试从 companion block table、positions 和 batch indices 动态派生；capture synthetic input 缺少完整组合，于是 fail-closed：

```text
rtp_llm/models_py/modules/hybrid/indexer.py:148-175
```

模型配置本身已注明当前限制：独立 indexer slot mapping 由 host 计算，CUDA graph 尚不支持：

```text
rtp_llm/models/deepseek_v2.py:553-560
```

底层 `SparseMlaParams` 也明确拒绝 companion pool 的非 graph-persistent slot buffer：

```text
rtp_llm/models_py/bindings/cuda/SparseMlaParams.cc:434-470
```

## 为什么现有 eager fallback 不够

`CudaGraphRunner::canRun()` 已支持因为 batch 超范围、graph key 缺失、tag 不匹配或 position id 不兼容而回退 eager：

```text
rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc:744-873
```

但当前问题发生在服务启动 capture 阶段，不在 request-time `canRun()`。capture 异常会终止 rank，无法自动回退。

即使 capture 修好，`canRun()` 当前也不检查：

- C batch gate 的 `native_only/managed` 状态；
- block table 的 offload 0-sentinel；
- admission generation；
- batch 中是否存在 offloaded row。

C 的 Python gate 和 remap/fetch 在 capture 时执行一次，graph replay 不会重新执行 Python 分支。若 resident synthetic batch capture 成功，而后 managed/offloaded batch仍被判定可 replay，attention 可能绕过实时 lossless fetch，从 0-sentinel 对应的错误 main KV 位置读取数据。这属于静默错算风险。

## 阶段 3 判定

阶段 3 的前提是 A1 实测 graph 收益至少满足：

```text
≥2ms/token 或 ≥2%
```

A1 graph-on 无法启动，收益未知，因此门槛不通过。本任务不实现双模式。

这不是一个适合“先加开关试试”的小改动。最小安全设计至少包括：

1. 为 `dsa_indexer_k` companion table 和 slot mapping 建立 graph-persistent host/device buffer；
2. capture 和 replay 前正确刷新 companion block table、positions、batch indices 和 slot mapping；
3. 修正 MLA 路径的 `is_cuda_graph` 传播，确保底层使用一致的 graph contract；
4. 给 `CudaGraphRunner::canRun()` 增加引擎可见的 offload eligibility：明确 `native_only` 才能 replay，managed/unknown 强制 eager；
5. eligibility 不能依赖 capture 时执行一次的 Python布尔值；
6. managed eager 路径继续保持 fail-closed，不能因 graph 不兼容而降级为错误的原生 attention。

涉及模块至少包括 graph input refresh、MLA/indexer 参数构建、tagged companion cache 和 request-time eligibility，属于中等以上跨层改动，明显超出最小兼容补丁。应先单独设计并补 capture/replay 测试，再恢复收益测量。

## 限定

- 本任务没有得到 graph-on TPOT，不能声称 graph 无收益，也不能估算收益上限。
- C graph-on 未进入请求面，所以没有 graph-on hash；兼容性结论只到“启动 capture 失败”。
- 所有成功请求来自 graph-off 基线。
- 没有修改运行代码，没有提交。

## 证据

```text
v32_migration/data/cudagraph_20260908/
```

主要文件：

```text
A1_g0_P1/P2_*
A1_g0_M1/M2_*
A1_g1_attempt1_D_stdout.log
A1_g1_attempt2_D_stdout.log
A1_g1_attempt1_summary.txt
A1_g1_attempt2_summary.txt
C_g1_attempt1_D_stdout.log
C_g1_attempt2_D_stdout.log
C_g1_attempt1_summary.txt
C_g1_attempt2_summary.txt
reference_C_g0_P1/P2.jsonl
reference_C_g0_M1/M2.jsonl
```
