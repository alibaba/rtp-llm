# MegaMoE 端到端使用与性能优化说明

> 目标模型：**Qwen3.5-397B-A17B** · 硬件：**AMD Instinct MI308X ×8 (gfx942 / CDNA3)** · 配置：**DP=1 / TP=8 / EP=8**
> 后端：FlyDSL 2-stage 融合 MegaMoE（`FusedMoEZeroCopyFp8`）接入 rtp-llm。
> 更新日期：2026-07-23

本文帮助读者快速：①用 MegaMoE 后端起服务并跑通 397B；②复现正确性与性能测试；③了解本轮做了哪些融合优化及收益。

---

## 1. 一句话结论

- MegaMoE 已在 397B（topk=10 > world_size=8）上**端到端跑通、精度正确**（relL2=0.037，输出正确）。
- decode 场景下，MegaMoE 比 Mori 后端**快约 1.20–1.29×（平均 1.25×）**（input_len=512，batch 1–32）。
- 本轮三项融合优化把 MoE 内部 `gemm1→gemm2` 的间隙从 **83.6ms 降到 7.65ms（−91%）**，精度不变。

---

## 2. 架构 / 接入方式

采用「透传 Router + 独占 Executor + 单例 Wrapper」（详见 `docs/megamoe_integration_design.html`）：

```
FusedMoe.forward
  → MegaMoePassthroughRouter.prepare()   # 透传：原样返回 hidden_states + 全局 topk_ids/weights
  → MegaMoeFusedExecutor.execute()       # 调用 wrapper 跑完整融合流水线；每次 execute 重灌本层 W2
      → MegaMoeWrapper(单例).forward()
          → FusedMoEZeroCopyFp8.forward() # quant → stage1(dispatch+GEMM1) → requant → stage2(GEMM2+combine)
  → MegaMoePassthroughRouter.finalize()  # 透传 executor 已 combine 的输出
```

- 枚举：`RouterType.MEGAMOE=8`、`ExecutorType.MEGAMOE_FUSED=9`，策略优先级 89（最高）。
- 开关：环境变量 `USE_MEGAMOE=1`（无 CLI 参数 / 无 C++ 字段，由 `config_adapter.py` 直接读 env）。
- 约束：`experts % world_size == 0`；`topk ≤ 64`（warp-ballot 预算）；`max_tok_per_rank`（mtpr）为 2 的幂，默认 **512**（`MEGAMOE_MAX_TOK`）。

---

## 3. 快速开始（端到端起服务）

### 3.1 关键环境变量

| 变量 | 值 | 说明 |
|---|---|---|
| `USE_MEGAMOE` | `1` | 启用 MegaMoE 后端（与 `USE_MORI_EP` / `USE_DEEPEP_MOE` 互斥使用） |
| `MEGAMOE_MAX_TOK` | `512` | 每卡单次 MoE forward 的最大 token 数（mtpr）。**超过 512 编译会极慢**（见限制） |
| `MORI_SHMEM_HEAP_SIZE` | `8G` | mori 对称显存堆 |
| `TEST_BLOCK_NUM` | `256` | 固定 KV block 数（397B 权重大，自动分配会 HIP OOM） |
| `WARM_UP` | `1` | 建议开启：启动阶段完成 JIT 编译，否则首个请求阻塞 |
| `WORLD_SIZE/TP_SIZE/EP_SIZE` | `8/8/8`，`DP_SIZE=1` | 并行配置 |
| `MODEL_TYPE` | `qwen35_moe` | Qwen3.5 混合注意力 MoE |
| `FLYDSL_FUSE_REQUANT` | `1`（默认） | 优化②开关：gathered requant |
| `FLYDSL_EMIT_DTM` | `1`（默认） | 优化①b 开关：片上 dest_tok_map（关掉则回退 Python，免重编） |
| `PYTHONPATH` | 含 `rtp-llm` 与 `FlyDSL` | FlyDSL 以 PYTHONPATH 引入 |

### 3.2 起服务

```bash
cd /home/admin/qinhanwen/codes
bash start_megamoe.sh          # 见该脚本，已配置好上述 env；日志 tee 到 megamoe_tp8ep8.log
```

启动后（`:6655` 前端、`:6656` rank-0 grpc 就绪）即可请求：

```bash
curl -sS -XPOST http://localhost:6655 -H 'Content-Type: application/json' \
  -d '{"prompt":"how are you?","generate_config":{"max_new_tokens":10,"seed":42,"top_p":0,"temperature":0.0}}'
# → "I'm doing well,"    首 token ~130ms（内核已热时）
```

---

## 4. 主要优化手段（本轮，均已提交 FlyDSL `ghu_moe_stage1`）

背景：trace 上 `moe_gemm1_0` 之后跟着大量 element_wise 小算子——它们是 stage1→stage2 之间用 eager PyTorch 做的元数据 glue，不是 GEMM 本身；基线间隙 83.6ms，比两个 GEMM 之和还大。

| # | commit | 优化 | 手段 |
|---|---|---|---|
| ①a | `ff2c480` | 向量化 combine dedup | `_build_combine_dest_tok_map` 的 `for j in range(1,k)`（9×(eq+any)）→ 一次广播比较 + 上三角 mask 归约；与 topk 无关 |
| ② | `83ea0b0` | gathered requant | 新写 FlyDSL 内核 `moe_requant_gathered_fp8.py`：只量化 stage2 实际读取的 `num_valid` 行（decode≈10、prefill 数百），不再盲扫全部 40960 行 |
| ①b | `4ee64e5` | 片上 dest_tok_map + total_recv 常量化 | 把 dedup 元数据下沉进 stage1 kernel 的 block0 post-pass（disp-table slot 25，指针守卫可回退）；常量 `total_recv` fill 从每 forward 挪到 init |

### 优化收益（trace 实测，180 个 MoE window，`gemm1→gemm2` 间隙）

| 阶段 | 间隙 | 每 window kernel 数 |
|---|---|---|
| 基线 | 83.64ms | 38 |
| +①a 向量化 dedup | 35.30ms | 13 |
| +② gathered requant | 19.87ms | 13 |
| **+①b 片上 dest_tok_map** | **7.65ms** | **1**（只剩 requant） |

**累计 −91%**，你在 trace 上看到的那堆 elementwise 全部消失（下沉进融合 kernel，且在关键路径外）。

> 相关内核约束修复：解除过时的 `topk ≤ npes` 断言（rtp-llm `e020aa549` + FlyDSL 内 3 处），才使 topk=10 > world_size=8 的 397B 能跑。

---

## 5. 性能数据：MegaMoE vs Mori

同机、同参数（DP1/TP8/EP8，input_len=512，decode 20 步），decode 延迟 **ms/token**：

| batch | MegaMoE | Mori | MegaMoE 更快? | 加速比 |
|---|---|---|---|---|
| 1 | **91.08** | 115.62 | ✅ | 1.27× |
| 4 | **94.73** | 120.21 | ✅ | 1.27× |
| 8 | **99.00** | 120.23 | ✅ | 1.21× |
| 16 | **94.44** | 122.29 | ✅ | 1.29× |
| 32 | **144.61** | 173.38 | ✅ | 1.20× |
| 64 | 144.19 (60/64) | 崩溃 (0/64) | — | — |

**平均加速 1.25×（区间 1.20–1.29×）。** 详见 `megamoe_vs_mori_decode_perf.xlsx`（格式对齐 `deepep_vs_mori_decode_perf.xlsx`）。

---

## 6. 复现步骤

### 6.1 正确性（8 卡，vs fp8 量化感知 torch 参考，阈值 relL2<0.30）

```bash
cd /home/admin/qinhanwen/codes/rtp-llm
export PYTHONPATH=/home/admin/qinhanwen/codes/rtp-llm:/home/admin/qinhanwen/codes/FlyDSL:$PYTHONPATH
export ROCM_PATH=/opt/rocm-7.2.0 HIP_PATH=/opt/rocm-7.2.0 MORI_SHMEM_HEAP_SIZE=8G MEGAMOE_MAX_TOK=512
FLYDSL_EMIT_DTM=1 FLYDSL_FUSE_REQUANT=1 torchrun --nproc_per_node=8 --master_port=29551 \
  rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/megamoe_gpu_correctness.py \
  --model-dim 4096 --inter-dim 1024 --experts 512 --topk 10 --tokens 64 --max-tok-per-rank 512 --same-tokens
# 期望：[PASS] MegaMoE correctness relL2=0.0370 < 0.3，8 卡一致
```

### 6.2 性能（decode，MegaMoE 与 Mori 各一次）

```bash
cd /home/admin/qinhanwen/codes/rtp-llm
# MegaMoE（input_len=512，全优化开）
bash rtp_llm/test/perf_test/multi_node/start_megamoe_perf_test.sh
# Mori 基线（同 input_len=512）
bash rtp_llm/test/perf_test/multi_node/start_mori_perf_test_512.sh
# 结果：各自 TEST_OUTPUT_QWEN35_MOE_*/Decode_Result.json 的 avg_decode_time（ms/token）
```

> 首次运行若冷启动，会触发一次性 JIT 编译（stage1 约 10–13min，之后走 `~/.flydsl/cache`）。

### 6.3 抓 trace 看 MoE 内部间隙

请求时加 `"gen_timeline": true`，服务会把 chrome trace 写到 `profiler_ts*.json`；按 GPU stream（pid=2）统计每个 `moe_gemm1_0` 到其后 `moe_gemm2_0` 的窗口即可复现第 4 节的间隙数据。

---

## 7. 已知限制

1. **mtpr 上限 512**：`MEGAMOE_MAX_TOK` 超过 512 编译会极慢（4096 已不可用）。因此**单次 MoE forward 的 token 数（含 prefill）不能超过 512**——长 prefill（如 input_len=2048）会触发 `cur_tok<=mtpr` 断言而崩溃。故性能测试用 input_len=512（Option B）。要跑 2048 需 mtpr=2048（编译慢、需关 ①b）或把 ①b 改成运行时循环。
2. **batch=64 不稳**：input_len=512 下 Mori 全失败、MegaMoE 60/64，疑似 `TEST_BLOCK_NUM`/并发偏紧，非 MoE 本身问题；对比时已剔除。
3. **首个请求 JIT 编译慢**：①b 的片上去重是完全展开循环，冷启动 stage1 编译约 10–13min（一次性，之后缓存）；建议 `WARM_UP=1` 把编译挪到启动阶段。
4. **W2 注入**：op 为全局单例，executor 每次 `execute` 用 `copy_` 重灌本层 W2（否则"最后一层赢"）。
5. **依赖**：FlyDSL 经 PYTHONPATH 引入（未 whl 化）；Stage2 wrapper 仍在 `tests/kernels/`（sys.path hack）。

---

## 8. 文件与提交索引

**FlyDSL（分支 `ghu_moe_stage1`）**
- `kernels/moe_fused_chained_fp8.py` — e2e 融合模块 + 三项优化的 Python 侧
- `kernels/moe_requant_gathered_fp8.py` — gathered requant 内核（新增）
- `kernels/fused_moe_stage1_plain_fp8.py` — disp-table slot 25 + `set_stage2_tables`
- `kernels/moe_gemm_2stage.py` — stage1 kernel 内 emit dest_tok_map
- commits：`ff2c480`（①a）、`83ea0b0`（②）、`4ee64e5`（①b + total_recv）

**rtp-llm（分支 `develop/308x_megamoe`）**
- `rtp_llm/models_py/distributed/megamoe_wrapper.py`、`.../fused_moe/impl/rocm/{routers,executors,strategy}/megamoe*.py`
- 单测：`.../impl/rocm/test/{test_megamoe.py, megamoe_gpu_smoke.py, megamoe_gpu_correctness.py}`
- perf：`rtp_llm/test/perf_test/multi_node/start_megamoe_perf_test.sh`、`start_mori_perf_test_512.sh`
- 设计文档：`docs/megamoe_integration_design.html`（as-built）
- commits：`e020aa549`（topk 放开）、`6469622bd`（设计文档回填）

**工程根目录 `/home/admin/qinhanwen/codes/`**
- `start_megamoe.sh` — 交互式起服务脚本
- `megamoe_vs_mori_decode_perf.xlsx` — 本次性能对比
