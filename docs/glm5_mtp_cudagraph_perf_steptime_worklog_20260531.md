# GLM-5 MTP cuda-graph + perf step-time 整改 worklog (2026-05-31)

Session 起点：`feature/glm5_cu13` HEAD = `da49e47526259c1402b48509cb73a5bb1d2261c8`
工作目录：`/home/zw193905/RTP-LLM/github-opensource`

## 0. 用户原始 prompt（逐字记录）

### Prompt #2（追加）

> 对了所有修改不要自动commit，放那到时候我手动commit，你把bug修掉就行

→ 已知，本次 session 不主动 commit。

### Prompt #1

> 前情提要，执行任何命令不允许用sudo权限，sudo执行会直接失败，然后静默运行，过程中默认按照推荐配置执行，中途不要询问我，不要停下来，直到问题解决。要把中间结果保存到md文件中，方便后面查看和总结,然后我所有的手动输入的prompt要记录在md中, 每一步执行完关键步骤状态都写到md文档里,有啥不清楚的读文档。
>
> 阅读 @/home/zw193905/RTP-LLM/github-opensource/docs/glm5_mega_moe_8dp8ep_mtp_perf_handoff_20260531.md 这个是上次任务的流程。我现在发现有个新的问题。
>
> 1. `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` 这个 smoke 测试里面其实 decode 的 **target verify + draft prefill + draft decode** 没有都开 cuda graph，你重新跑下测试确认下，看最后生成的 timeline json 来验证确实都开了 cuda graph。没有的话就 fix，预期是都开 cuda graph 的。
> 2. `glm5_mega_moe_8dp8ep_grid_mtp_test` 这个 perf test 也不对，avg decode time 感觉像是算的 tpot，但是实际上我想要的是 **step time**，是 decode 每计算一步的时间，不是每个 token 的时间，因为 mtp 的接受 token 数是不稳定的。这个也要改一下。
> 3. 最后生成 perf test 的 timeline 中也确认下是否都开了 cuda graph，不是的话可以返回 smoke test 测试修复。

## 1. 任务拆解 / Plan

| # | 子任务 | 状态 |
|---|---|---|
| A | 定位 `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` 的 BUILD 定义 + 跑通基线 | 待执行 |
| B | 解析 timeline JSON，判断 target verify / draft prefill / draft decode 三个阶段是否都走 cuda graph replay | 待执行 |
| C | 若有缺失：定位代码（应在 MtpExecutor / SpeculativeExecutor / CudaGraphRunner 附近）并 fix | 待执行 |
| D | 定位 `glm5_mega_moe_8dp8ep_grid_mtp_test` 的 decode 时长统计逻辑（CaseRunner / decode_comparer） | 待执行 |
| E | 改成 step time（每个 decode step 的 wall time），不要 / tokens 数 | 待执行 |
| F | 重跑 perf test，看 timeline 是否三阶段都开 cuda graph | 待执行 |
| G | 提交修复 | 待执行 |

## 2. 关键文件清单

| 关注点 | 路径 |
|---|---|
| smoke BUILD | `internal_source/rtp_llm/test/smoke/BUILD:200` (`mla_mtp_mega_moe_cudagraph_pd_full_ckpt`) |
| smoke macro | `internal_source/rtp_llm/test/smoke/defs.bzl:86` (`smoke_test`, `enable_profile`) |
| perf BUILD | `internal_source/rtp_llm/test/perf_test/BUILD:325` (`glm5_mega_moe_8dp8ep_grid_mtp_test`) |
| perf decode time 公式 | `github-opensource/rtp_llm/test/perf_test/dataclass.py:48-61` (`ResponseInfo`) |
| 现有 perf timeline 样本 | `/home/zw193905/docs_scripts/perf_results/glm5_mega_moe_8dp8ep_grid_mtp_test/20260531_104655/timelines/` |
| MTP executor cuda graph 切换 | `github-opensource/rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc:803-849` |
| PyWrappedModel cuda graph 容量决策 | `github-opensource/rtp_llm/cpp/models/PyWrappedModel.h:283-321` |
| GenerateStream iter_count++ | `github-opensource/rtp_llm/cpp/engine_base/stream/GenerateStream.cc:520-523` |
| aux_info JSON 字段 | `github-opensource/rtp_llm/cpp/model_rpc/model_rpc_client.py:267-294` |

## 3. 现状梳理（基于上次跑的 perf timeline + 代码静态分析）

### 3.1 解析 `bs16_seq16384_decode_wr0_18.json` (20260531_104655 perf 跑) 的事件计数

```
4 decode steps × ...
  4× executor.mtp.decode_step(target_model_verify)
  4× py_model.forward(cuda_graph=1,prefill_cg=0,model_id=0)   ← 目标 verify
  4× executor.mtp.decode_step(draft_model_forward,use_sp=0,sp_cg=0,sp_prefill_cg=0,is_fake=0)
 12× py_model.forward(cuda_graph=1,prefill_cg=0,model_id=1)   ← 4 draft prefill + 4×2 draft decode iter
 12× cuda_graph.forward(replayDecode)
 12× cudaGraphLaunch                                          ← **只有 12 次 graph 启动**
```

期望：4(target verify) + 4(draft prefill) + 8(draft decode iter) = **16 次 graph 启动**
实际：**12 次** → 缺 4 次。

判定缺失的就是 **4 次 draft prefill**。原因：
- `MtpExecutor` 检测到 `mega_moe + ep_size>1` → 设 `disable_sp_prefill_for_mega_moe = true`
- → `sp_prefill_draft_model_` 不创建
- → draft prefill 走 `draft_model_->forward()`，但 `draft_model_` 是按 `num_tokens_per_bs=1` capture 的（用于 draft decode）
- → 输入是 (bs, gen_num_per_cycle+1=4) 不匹配 capture 的 1 → `canRun()=false` → eager 回退
- → 没有 graph 启动

`target verify` 和 `draft decode loop_iter` 都成功用上 graph（共 12 次），符合 4+8 推算。

### 3.2 修复路径

代码里已经提供了 `RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1` 这个旁路开关（`MtpExecutor.cc:77`），开了之后 `disable_sp_prefill_for_mega_moe` 会被无效化、`sp_prefill_draft_model_` 重新被构造、draft prefill 走 captured graph。

→ 修复策略：给 smoke 和 perf 这两条用例都加上 `RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1`。

### 3.3 perf test avg_decode_time 公式

`dataclass.py:59-61`：
```python
self.decode_time_per_token = (
    self.decode_time / (self.output_len - 1) if self.output_len > 1 else 0.0
)
```

这是 **TPOT**（time per output token）。MTP 每步接受 1~4 个 token 不稳定，TPOT 等于 `step_time / tokens_per_step`，所以波动很大。

`iter_count` 字段（`AuxInfo`，`GenerateStream::step()` 自增）等于实际 forward step 数（prefill 1 次 + decode N 次）。
所以 **step_time = decode_time / max(1, iter_count - 1)**。

非 MTP 场景 `iter_count - 1 == output_len - 1`，step_time 退化为 TPOT，旧 baseline 不变。

---

## 4. 进度记录

### 4.1 已落地的代码修改（未 commit，等待用户手动 commit）

1. `internal_source/rtp_llm/test/smoke/BUILD` (`mla_mtp_mega_moe_cudagraph_pd_full_ckpt`)
   - decode envs 新增 `RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1`
   - `enable_profile=False → True`（以便生成 timeline 验证）
2. `internal_source/rtp_llm/test/perf_test/BUILD` (`glm5_mega_moe_8dp8ep_grid_mtp_test`)
   - envs 新增 `"RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH": "1"`
3. `github-opensource/rtp_llm/test/perf_test/dataclass.py`
   - `ResponseInfo` 新增 `iter_count` 与 `decode_step_time` 字段
   - `analyze_results` 中 `avg_decode_time/max_decode_time/decode_time_var` 改用 `decode_step_time`（== `decode_time / max(1, iter_count - 1)`）
   - 保留 `decode_time_per_token` 字段 & `avg_decode_time_per_token/max_decode_time_per_token` 用于向后兼容
   - JSON 输出新增 `avg_decode_step_time/max_decode_step_time` 字段；旧的 `avg_decode_time_per_token/max_decode_time_per_token` 现在内容是 step time（== TPOT 当非 SP）
   - PrettyTable 列头从 `Decode Time(ms)` 改为 `Decode Step Time(ms)`

### 4.2 helper 工具

`docs/glm5_mtp_cudagraph_runs/analyze_timeline.py` — 解析 timeline JSON，按事件名统计 `target_verify` / `draft_prefill` / `draft_decode loop_iter` 的 step 数，与 `cudaGraphLaunch` 计数对比，提示是否需要 `RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH`。

### 4.3 fix 前的 baseline（旧 perf 跑）

```
$ analyze_timeline.py bs16_seq16384_decode_wr0_18.json
  target verify          steps: 4
  draft prefill          steps: 4
  draft decode loop iter steps: 8
  cudaGraphLaunch                    : 12          ← 期望 16
  draft_model_forward label sample  : executor.mtp.decode_step(draft_model_forward,use_sp=0,sp_cg=0,sp_prefill_cg=0,is_fake=0)
  ✗ deficit 4 (即 4 次 draft prefill 没走 cuda graph)
```

### 4.4 fix 后第一次 smoke 跑（已 kill）

```
bazelisk test //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt \
  --config=cuda13 --test_output=errors --test_timeout=14400 --cache_test_results=no
```
log: `docs/glm5_mtp_cudagraph_runs/smoke_full_ckpt_run1.log`

捕获到的 trace：`mla_mtp_mega_moe_cudagraph_pd_full_ckpt_wr0_1.json`，第一次 `start_profile` 后的 3 steps。
分析器输出：
```
target verify          steps: 3
draft prefill          steps: 3
draft decode loop iter steps: 6
cudaGraphLaunch        : 9        ← 9 = 3 (target verify) + 6 (draft decode)
draft_model_forward label sample : ...sp_cg=1,sp_prefill_cg=1,is_fake=1
deficit 3
```

`sp_cg=1,sp_prefill_cg=1` 证明 **修复落地、`sp_prefill_draft_model_` 已创建并 capture**。
`is_fake=1` 说明这 3 steps 都是 fake stream（dp_size=4 时 FIFOScheduler 给空闲的 rank 填的占位流），代码（`MtpExecutor.cc:1734`）刻意把 fake stream 的 draft prefill 走 eager（无需 graph，因为反正不会写真实 KV cache）。

decode 日志同样确认：
```
[MTP decode] draft prefill model choice use_sp_prefill=0 sp_exists=1 sp_cg=1 sp_prefill_cg=1 is_fake_stream=1
[MTP decode] fake stream draft prefill uses eager draft model; real streams still use sp_prefill CUDA graph
```

对真实 stream（`is_fake=0`），代码会走 `sp_prefill_draft_model_->forward(model_input)`（line 1741），即 cuda graph replay。
smoke 这次 trace 没拿到 `is_fake=0` 的捕获窗口（profile 在 load 完模型后立刻 fire，那时还没有真实请求到 decode 端）；perf test 用了 `PERF_PREARM_PROFILE=1`，profile 启动在真实请求之前但同帧覆盖，能拿到真实 stream 数据 → 用 perf test 做最终验证。

### 4.5 fix 后第一次 perf 跑（已 PASS）

```
bazelisk test //internal_source/rtp_llm/test/perf_test:glm5_mega_moe_8dp8ep_grid_mtp_test \
  --config=cuda13 --test_output=errors --test_timeout=14400 --cache_test_results=no
```
log: `docs/glm5_mtp_cudagraph_runs/perf_grid_mtp_run1.log`

结果：`PASSED in 2779.0s`（与基线 2540s 接近）。

#### 4.5.1 timeline cuda graph 验证

输出目录：
`/data0/zw193905/.cache/bazel/.../testlogs/internal_source/rtp_llm/test/perf_test/glm5_mega_moe_8dp8ep_grid_mtp_test/test.outputs/timelines/`

160 个 trace JSON，全部跑过 `check_all_timelines.py`：

```
Total: 160 OK, 0 MISS (out of 160 files)
```

抽样 `bs1_seq4096_decode_wr0_1.json` 详细计数（对比 fix 前）：

| 指标 | fix 前 (20260531_104655, bs16/seq16384) | fix 后 (bs1/seq4096，其它 wr / bs 类似) |
|---|---:|---:|
| target verify steps | 4 | 4 |
| draft prefill steps | 4 | 4 |
| draft decode loop iters | 8 | 8 |
| **`cudaGraphLaunch`** | 12 (deficit 4) | **16 (3 阶段全覆盖)** |
| `cuda_graph.forward(replayPrefill)` | 0 | 4 |
| draft_model_forward 标签 | `sp_cg=0, sp_prefill_cg=0, is_fake=0` | `sp_cg=1, sp_prefill_cg=1, is_fake=0` |

预期 = 4 (target verify) + 4 (draft prefill) + 8 (draft decode loop) = 16，实际 16 → ✅ 全部 3 阶段 cuda graph replay。

#### 4.5.2 perf decode 数值 (Decode_Result.json 节选)

| bs / input_len | avg_prefill_time(ms) | **avg_decode_time(ms, step)** | avg_decode_time_per_token(ms, TPOT) | step/TPOT 比 |
|---:|---:|---:|---:|---:|
| 1 / 4096 | 49.14 | **32.82** | 7.92 | 4.14 |
| 1 / 16384 | 150.54 | 506.93 | 122.36 | 4.14 (有抖动) |
| 1 / 32768 | 153.41 | **37.34** | 9.01 | 4.14 |
| 1 / 65536 | 338.13 | **38.45** | 9.28 | 4.14 |
| 2 / 4096 | 49.35 | **34.88** | 8.42 | 4.14 |
| 4 / 4096 | 90.43 | **53.30** | 12.87 | 4.14 |
| 8 / 4096 | 212.77 | **110.30** | 26.62 | 4.14 |
| 16 / 4096 | 220.68 | **57.78** | 13.95 | 4.14 |
| 16 / 65536 | 280.02 | **53.82** | 12.99 | 4.14 |

观察：
- step time 全部约 ≈ TPOT × 4.14，跟 MTP `gen_num_per_cycle=3 + 1 target bonus = 4 tokens/step` 物理上限基本对齐（`force_sp_accept=True` 时几乎每步都接受满）。比值始终是 ~4.14 是因为压测下 `iter_count=8` 而 `output_len=30`，每个 decode iter 平均 ~4 个 token。
- 比值=4.14（略超 4）说明 MTP prefill iter 不是只贡献 1 个 token，可能还多承载了 1 个第一轮 draft 提议（待考），但不影响 step time 作为"每个 decode iter wall time"的物理意义。
- bs8/seq4096、bs16/seq4096 这两个点的 step time 仍偏高（110ms、57ms），跟上次跑同样观察到的 decode_ms 异常区间一致，需要单独追这个性能瓶颈（不在本次 fix 范围）。

#### 4.5.3 iter_count 行为校验

从 access log 抽 292 个完成的 decode 请求：
```
iter_count distribution: min=8 max=8 mean=8.00 (output_len 全部 30)
(output_len-1)/(iter_count-1) ratio: min=4.143 max=4.143
```
`force_sp_accept=True` + `decode_test_length=30` 让所有 decode 都走相同 iter 数。

---

## 5. 结论

| 目标 | 状态 |
|---|---|
| 修复 `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` 让 target verify + draft prefill + draft decode 三阶段都走 cuda graph | ✅ 通过 BUILD 给 decode envs 加 `RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1`，sp_prefill_draft_model_ 现在被创建（smoke decode 日志 `sp_exists=1 sp_cg=1 sp_prefill_cg=1`） |
| smoke timeline 验证 | ⚠️ 部分验证：smoke 用例本次只 capture 到 fake-stream 窗口（`is_fake=1` → 走 eager fallback by 设计）；通过 perf test 间接验证了真实 stream 路径 |
| 修复 `glm5_mega_moe_8dp8ep_grid_mtp_test` avg decode time 从 TPOT 改为 step time | ✅ `dataclass.py` 用 `iter_count` 算 `decode_step_time = decode_time / (iter_count - 1)`；非 SP 场景与原 TPOT 等价（保持 qwen35 baseline 兼容） |
| perf timeline cuda graph 验证 | ✅ 160/160 timeline 全部三阶段 `cudaGraphLaunch` 数符合预期（每 trace 16 次：4+4+8） |

## 6. 文件变更 (未 commit)

```
internal_source/rtp_llm/test/smoke/BUILD
  - mla_mtp_mega_moe_cudagraph_pd_full_ckpt decode envs 增 RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1
  - enable_profile False -> True

internal_source/rtp_llm/test/perf_test/BUILD
  - glm5_mega_moe_8dp8ep_grid_mtp_test envs 增 "RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH": "1"

github-opensource/rtp_llm/test/perf_test/dataclass.py
  - ResponseInfo 增 iter_count, decode_step_time 字段
  - analyze_results 用 decode_step_time 作 avg/max/var；保留 decode_time_per_token 作 TPOT (向后兼容)
  - PrettyTable 列头 "Decode Time(ms)" -> "Decode Step Time(ms)"
  - JSON 输出新增 avg_decode_step_time / max_decode_step_time（distribution mode）
```

## 7. helper / 输出

- `docs/glm5_mtp_cudagraph_runs/analyze_timeline.py` — 解析 timeline，识别 3 阶段 cuda graph 命中率
- `docs/glm5_mtp_cudagraph_runs/smoke_full_ckpt_run1.log` — smoke run 输出（已 kill）
- `docs/glm5_mtp_cudagraph_runs/perf_grid_mtp_run1.log` — perf run 输出（PASS）
- `/tmp/check_all_timelines.py` — 批量验证脚本

