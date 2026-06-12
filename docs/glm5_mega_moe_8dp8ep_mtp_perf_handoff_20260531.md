# GLM-5 mega-MoE 8DP8EP MTP perf 数据归档 + handoff (2026-05-31)

本文档记录 2026-05-31 这次 session 的全流程：用户任务、测试命令/环境/结果、数据归档操作、git 提交，供新 session 在此基础上接着推进。

---

## 1. 用户输入任务

> 把 `/home/zw193905/docs_scripts` 下的
> `glm5_mega_moe_8dp8ep_grid_mtp_test_20260531_attempt2` 和
> `perf_results/glm5_mega_moe_8dp8ep_grid_prefill_mtp_test`
> 的数据归类到一个文件夹里，然后 commit 并 pull。

确认细节后用户选定方案：

- **目录结构**：只把 `attempt2` 归入已有 decode 目录，prefill 目录不动。
- **timelines 处理**：不 tar，直接提交 raw JSON。
- **Git 操作**：commit + push（不是 pull）。

---

## 2. 涉及的两个 Bazel 测试

两个测试都在内源仓 `internal_source/rtp_llm/test/perf_test/`。

### 2.1 Decode：`glm5_mega_moe_8dp8ep_grid_mtp_test`

```bash
bazelisk test //internal_source/rtp_llm/test/perf_test:glm5_mega_moe_8dp8ep_grid_mtp_test \
  --config=cuda13 --test_output=errors --test_timeout=14400 --cache_test_results=no
```

### 2.2 Prefill：`glm5_mega_moe_8dp8ep_grid_prefill_mtp_test`

```bash
bazelisk test //internal_source/rtp_llm/test/perf_test:glm5_mega_moe_8dp8ep_grid_prefill_mtp_test \
  --config=cuda13 --test_output=errors --test_timeout=14400 --cache_test_results=no
```

两者都是 grid 模式，结果通过 `CaseRunner` 启服务、跑 batch_size × input_len 网格、写出 `*_Result.json` + `timelines/` profiler trace。

---

## 3. 本次纳入归档的两批数据

### 3.1 Decode attempt2（本次重点）

| 字段 | 值 |
|---|---|
| 原路径 | `/home/zw193905/docs_scripts/glm5_mega_moe_8dp8ep_grid_mtp_test_20260531_attempt2/` |
| 新路径 | `/home/zw193905/docs_scripts/perf_results/glm5_mega_moe_8dp8ep_grid_mtp_test/20260531_104655/` |
| 测试时间 | 2026-05-31 10:46:55 CST |
| Git commit (被测) | `cdc1b18b69b79cebe06b4797914d3ff2f42251b9`（RTP-LLM 内源） |
| Bazel 结果 | `PASSED in 2540.9s` |
| 拓扑 | `tp_size=1`, `dp_size=8`, `ep_size=8`, `world_size=8` |
| 压力参数 | `max_seq_len=131072`, `concurrency_limit=16`, `enable_cuda_graph=1`, `gen_num_per_cycle=3`, `sp_type=eagle`, `sp_model_type=glm_5_mtp` |
| 失败/异常 | 无 `SIGSEGV` / `SIGABRT` / CUDA OOM/error / DeepGEMM timeout / scheduler-shape `ValueError` / coredump |
| 唯一观察 | profiler `External init callback must run in same thread as registerClient`（不影响运行，所有 metric `success_rate=1.0`） |
| 环境变量备注 | 本次最终 run 没有 `CUDA_LAUNCH_BLOCKING`，也没有 `DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS` |

#### 3.1.1 测试 grid 配置（来自 `test_info.json`）

```json
{
  "model_type": "glm_5",
  "checkpoint_path": "/home/zw193905/models/GLM-5-BF16",
  "tokenizer_path": "/home/zw193905/models/GLM-5-BF16",
  "tp_size": "1",
  "dp_size": 8,
  "max_seq_len": 131072,
  "concurrency_limit": 16,
  "decode_test_length": 30,
  "num_measures": 10
}
```

#### 3.1.2 Decode 结果（`Decode_Result.json` 全部 20 行 grid 点位 ms）

batch × input_len 全部 `success_rate=1.0`，`avg_wait_time=0`。

| batch \ input_len | 4096 | 16384 | 32768 | 65536 |
|---:|---:|---:|---:|---:|
| **bs=1 prefill_ms / decode_ms** | 42.23 / 7.91 | 77.83 / 9.26 | 165.04 / 9.53 | 174.32 / 9.83 |
| **bs=2 prefill_ms / decode_ms** | 301.47 / 8.82 | 154.08 / 9.31 | 137.80 / 9.49 | 345.22 / 9.85 |
| **bs=4 prefill_ms / decode_ms** | 55.33 / 13.53 | 136.32 / 21.26 | 167.11 / 10.02 | 250.04 / 10.27 |
| **bs=8 prefill_ms / decode_ms** | 377.25 / 14.54 | 92.49 / 10.15 | 142.59 / 10.79 | **888.11 / 100.21** |
| **bs=16 prefill_ms / decode_ms** | 633.28 / **53.36** | 111.18 / 11.30 | **668.20 / 102.50** | 336.61 / 13.57 |

**注意**：bs8/seq65536、bs16/seq4096、bs16/seq32768 这三个点 decode_ms 异常飙到 50~102 ms，正常应在 10~14 ms 区间——这是新 session 需要重点关注/复现/复盘的异常点。

#### 3.1.3 Timelines

- 160 个 trace JSON：`timelines/bs{1,2,4,8,16}_seq{4096,16384,32768,65536}_decode_wr{0..7}_{step}.json`
- 总大小约 2.5 GB（每个 ~19 MB）
- 命名规律：`wr0..wr7` 对应 8 个 dp 实例；尾缀 step 编号
- 用 `gen_timeline` 或 Perfetto 打开查看 GPU 时序

### 3.2 Prefill `20260529_153708`（顺带补提）

| 字段 | 值 |
|---|---|
| 路径 | `/home/zw193905/docs_scripts/perf_results/glm5_mega_moe_8dp8ep_grid_prefill_mtp_test/20260529_153708/` |
| 状态 | 只有 `timelines/`（32 个 raw JSON，495 MB），**没有** `*_Result.json` / `test_info.json` |
| 同目录另一时间戳 | `20260525_020952/` 是完整版（带 `Prefill_Result.json` + `report.md` + `timelines.tar.gz` 200 MB） |
| 用途 | 仅作为补充 trace 数据，用于跨次 prefill 对比 |

### 3.3 既有的 prefill 完整结果（`20260525_020952/`，仅供参考）

| input_len | bs | success | avg_prefill_ms |
|---:|---:|---:|---:|
| 4096 | 1 | 1.000 | 176.30 |
| 16384 | 1 | 1.000 | 467.57 |
| 32768 | 1 | 1.000 | 556.27 |
| 65536 | 1 | 1.000 | 1001.95 |
| 131072 | 1 | 1.000 | 2129.48 |

注：prefill 测试参数和 decode 不同——`tp_size=8`, `dp_size=1`, `concurrency_limit=1`，MTP 参数 `--sp_model_type glm_5_mtp --gen_num_per_cycle 3 --sp_type eagle --sp_checkpoint_path /home/zw193905/models/GLM-5-FP8 --sp_act_type bf16`。

---

## 4. 测试环境

| 项 | 值 |
|---|---|
| 模型 | GLM-5（BF16 ckpt：`/home/zw193905/models/GLM-5-BF16`；MTP draft 用 FP8 ckpt：`/home/zw193905/models/GLM-5-FP8`） |
| 平台 | NVIDIA CUDA 13（`--config=cuda13`） |
| Python | `/opt/conda310/bin/python3` |
| 构建 | Bazel (Bazelisk) |
| 仓库根 | `/home/admin/zw193905/RTP-LLM/`；本次工作在子模块 `github-opensource/`（分支 `feature/glm5_cu13`） |
| 内源 commit (decode 跑的版本) | `cdc1b18b69b79cebe06b4797914d3ff2f42251b9` |

确定性环境变量（smoke 默认就开，perf 同源）：`DETERMINISTIC_GEMM=1`、`ENABLE_STABLE_SCATTER_ADD=ON`。

---

## 5. 数据归档操作流水

按时间顺序执行的命令（在 `/home/zw193905/docs_scripts/` 仓库里，分支 `master`，remote `git@gitlab.alibaba-inc.com:zw193905/docs_scripts.git`）：

```bash
# 1. 物理移动 attempt2 到目标位置（时间戳来自 test_meta.json: 2026-05-31T10:46:55）
mv /home/zw193905/docs_scripts/glm5_mega_moe_8dp8ep_grid_mtp_test_20260531_attempt2 \
   /home/zw193905/docs_scripts/perf_results/glm5_mega_moe_8dp8ep_grid_mtp_test/20260531_104655

# 2. 仅 stage 这两个相关目录（避开 tmp/whl.txt 和 worklogs/）
cd /home/zw193905/docs_scripts
git add perf_results/glm5_mega_moe_8dp8ep_grid_mtp_test/20260531_104655/ \
        perf_results/glm5_mega_moe_8dp8ep_grid_prefill_mtp_test/20260529_153708/

# 3. .gitignore 自动排除 *.log，所以 test.log/process.log/attempt2.log 不会进 commit

# 4. commit
git commit -m "Add GLM-5 mega-MoE 8DP8EP MTP perf results (decode 20260531 + prefill 20260529) ..."
# → 196 files changed, 46,110,527 insertions(+)
# → commit 31ed0e1

# 5. push
git push origin master
# → 58bd363..31ed0e1  master -> master
```

### 5.1 仓库的归档惯例

- 路径 pattern：`perf_results/<test_target_name>/<YYYYMMDD_HHMMSS>/`
- 一个完整跑次包含：`<X>_Result.json` + `test_info.json` + `test_meta.json` + `logs/` + `timelines.tar.gz`（或 raw `timelines/`）+ 可选 `report.md`
- 本次 attempt2 没生成 `report.md`，但 `README.md` 起到了同样作用
- 现有 commit message 约定参见 `git log --oneline` 前几条（`Add GLM5 mega MoE perf results` / `feat: GLM-5 mega MoE 8DP8EP perf data ...`）

### 5.2 最终目录布局

```
perf_results/
├── glm5_mega_moe_8dp8ep_grid_mtp_test/                ← decode MTP
│   ├── 20260525_050830/                                  (原有，timelines.tar.gz 200MB)
│   └── 20260531_104655/                                  (本次新增，raw timelines/ 2.5GB)
│       ├── Decode_Result.json
│       ├── README.md
│       ├── test_info.json
│       ├── test_meta.json
│       └── timelines/      (160 个 raw JSON)
└── glm5_mega_moe_8dp8ep_grid_prefill_mtp_test/        ← prefill MTP
    ├── 20260525_020952/                                  (原有完整版)
    └── 20260529_153708/                                  (本次补提，仅 timelines/)
        └── timelines/      (32 个 raw JSON)
```

未 stage 的 untracked 残留（保留未动）：`tmp/whl.txt`、`worklogs/`。

---

## 6. 给下一个 session 的接力提示

1. **本次没动** 的两类数据：`tmp/whl.txt` 和 `worklogs/`，如果新任务涉及到，记得先看再决定。
2. **decode_ms 异常点** 值得复盘：bs8/seq65536（100 ms）、bs16/seq4096（53 ms）、bs16/seq32768（102 ms）。前后行同 batch 都是 10~14 ms，怀疑：
   - scheduler 切到非 cuda-graph fallback
   - MTP draft cuda graph 失效（参考 memory：`glm5-mtp-pd-cudagraph-investigation`）
   - 某些 seq 长度刚好踩到 bucket 边界 → 用 timelines/ 验证
3. **timelines 是 raw 形式入库** 的（用户本次明确选择），仓库会因此长大约 3 GB。如果以后还要继续提交类似数据，建议先 `tar czf timelines.tar.gz timelines/` 再 `git rm -r timelines/`（沿用 `20260525_050830` 的 pattern）。
4. **如果要本地重复实验**：进入 dev container 后
   ```bash
   cd /home/admin/zw193905/RTP-LLM/github-opensource
   bazelisk test //internal_source/rtp_llm/test/perf_test:glm5_mega_moe_8dp8ep_grid_mtp_test \
     --config=cuda13 --test_output=errors --test_timeout=14400 --cache_test_results=no
   ```
   结果会落在 perf_test 工作目录（具体路径可查 `internal_source/rtp_llm/test/perf_test/` 下的 runner）。
5. **当前 git 状态**（截至本文写入时）：
   - `docs_scripts`：分支 `master`，HEAD = `31ed0e1`，已 push 到 origin。
   - `RTP-LLM/github-opensource`：分支 `feature/glm5_cu13`，HEAD = `da49e4752 fix - fix batch error`，工作树有大量 untracked docs/分析文件（参见 session 启动时的 `git status`）——本次没有触碰内源/外源代码。

---

## 7. 关键引用

- 测试 macro 定义：`internal_source/rtp_llm/test/smoke/defs.bzl`（perf_test 同套）
- CaseRunner / Comparer：`internal_source/rtp_llm/test/smoke/case_runner.py`、`*_comparer.py`
- 项目说明：`/home/admin/zw193905/RTP-LLM/CLAUDE.md`
- 相关记忆条目：`glm5-mtp-implementation`、`glm5-mtp-pd-cudagraph-investigation`、`glm5-pd-prefill-execnoblockcopy-bug`
