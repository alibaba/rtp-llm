# P2P Smoke：Prefill CP2（KV 分片）→ Decode DP2

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_cp2_sharded_to_dp2_decode_entrance`
- 模型：GLM-5 FP8 4-layer（MLA）
- Prefill：CP2，开启 `prefill_cp_kv_cache_sharded=1`
- Decode：DP2 × TP1，显式声明 `prefill_cp_size=2`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证 MLA fixed-region/CP KV 分片从两个 Prefill CP rank 向两个 Decode DP 副本展开 route，覆盖 CP 字节切分、DP 复制与传输 key 一致性。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：请求和 P2P 传输成功，服务退出崩溃

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round1.log`，以及 Bazel `test.outputs/outputs.zip` 中的 `prefill_logs/process.log`、`decode_logs/process.log`
- 状态：`FAILED`，执行 1 个 Bazel test、1 个 query；Bazel 202.161 秒，test 180.0 秒

#### 请求与 P2P 证据

- Prefill/Decode 各 2 rank 均完成启动，四个 rank 均初始化 TCP `P2PConnector`。
- Prefill rank 0/1 各报告 `sent=4/4, all_cb_received=1, cancelled=0`。
- 请求返回 HTTP 200，actual 响应为 `acheridera NaughtyMZ Terminating賞RpcSGREEN`，与 golden 完全一致。
- 未出现 `no matching recv task`、`transfers not all done` 或 `P2P cache load failed`。

上述证据确认本轮真实覆盖了 MLA fixed-region CP 字节切片、两个 Prefill CP rank 到两个 Decode DP 副本的 P2P 传输，并完成了 Decode 推理。现有输入只有 1 个 query，所以虽然 target 配置了 `reuse_cache=1`，本轮不覆盖跨 query reuse。

#### 最终失败现象

请求比较完成后 runner 终止服务：

- Prefill rank 1 未在 10 秒内发出 shutdown-ready acknowledgement；随后 rank 0 在 CPython exit handler 中析构 `pybind11::function` 时触发 `PyThreadState_Get` GIL fatal 和 SIGABRT。
- Decode rank 1 收到 SIGTERM 后发生 SIGSEGV，rank 0 最终被 shutdown timeout 强杀。
- `MagaServerManager.stop_server()` 正确地把非干净退出判为失败，所以不能把本轮 Bazel 状态记为通过。

#### 判断与定位方向

这是已在 `p2p_mla_moe_smoke_validation_report.md` 记录的 DeepEP/多 rank 服务关闭缺陷再次出现，不是 TransferPlan、CP 字节切片、P2P 收发或 golden 问题。下一步对照同一 GLM-5 模型的 legacy/non-P2P target；若退出栈相同，则将其作为独立 shutdown 缺陷处理，并保留本场景的功能通过、target 失败双重结论。

### Round 2：legacy MLA 对照复现同一退出缺陷

- 日期：2026-08-30
- 对照 Target：`//rtp_llm/test/smoke:mla_cp_pd`（legacy cache-store，不启用 P2PConnector）
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round2_legacy_control.log`
- 状态：`FAILED`，执行 1 个 Bazel test、1 个 query；Bazel 196.215 秒，test 177.1 秒

legacy target 的请求同样返回 HTTP 200，actual 文本与 golden 一致；退出时同样出现 Prefill `PyThreadState_Get` GIL fatal、rank 1 shutdown-ready 超时，以及 Decode rank 1 SIGSEGV。由于对照完全不经过 P2PConnector，这一轮确认 Round 1 的最终 Bazel 失败与 TransferPlan/P2P 无关。

后续方向分成两条：本场景继续保留“P2P 请求与传输成功、target 因通用 shutdown 缺陷失败”的准确状态；另行评估可否使用不触发 DeepEP 析构缺陷、但仍保留 MLA + MoE + CP 字节分片语义的合法配置，使新增回归 target 能获得干净的 Bazel PASS。

### Round 3：shutdown request event 避免 GIL fatal，但仍超时

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round3_shutdown_event.log`
- 状态：`FAILED`，1 个 Bazel test、1/1 query

改用 multiprocessing event 让 rank 在 Python 主循环内进入 `BackendManager.stop()` 后，请求、P2P 与 golden 仍成功，且未再出现 SIGTERM 被 DeepEP 覆盖后的 GIL fatal/SIGSEGV；但 Decode 两 rank 未能在 50 秒内结束，仍被强杀。定位方向从“信号处理绕过清理”收窄到“多 rank engine/distributed teardown 顺序”。

### Round 4：提前 reset DeepEP 仍不足

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round4_deepep_reset.log`
- 状态：`FAILED`，1 个 Bazel test、1/1 query

在销毁 torch distributed 前增加 `DeepEPWrapper.reset()`，请求、P2P 与 golden 正常，但部分 rank 仍停在 `NormalEngine::stop()` 或 distributed destroy。说明 DeepEP buffer 释放是必要清理候选，但单独提前 reset 不能建立跨 rank 的退出同步。

### Round 5：event 广播单次通过

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round5_event_broadcast.log`
- 状态：`PASSED`，1 个 Bazel test、1/1 query；test 79.0 秒

leader/follower 在等待 acknowledgement 前同时收到 event，本轮完整退出。但这是一次时序窗口成功；后续 Round 7 在相同代码上再次失败，因此不能视为稳定修复。

### Round 6：legacy cache-store 对照继续复现

- 日期：2026-08-30
- 对照 Target：`//rtp_llm/test/smoke:mla_cp_pd`
- 日志：`build_logs/p2p_cp2_sharded_to_dp2_round6_legacy_shutdown_control.log`
- 状态：`FAILED`，1 个 Bazel test、1/1 query

legacy cache-store 请求和 golden 成功，low-latency EP 仍在 shutdown 阶段挂死。该对照再次确认根因不在 P2PConnector 或 TransferPlan。

### Round 7：event 实验最终矩阵再次失败

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`FAILED`，1 个 Bazel test、1/1 query；test 130.9 秒

P2P/golden 成功后 Decode rank 1 已进入 distributed destroy，rank 0 仍停在 `NormalEngine::stop()`，50 秒后两者被强杀。Round 5 的单次通过被证实为 flaky；同批另外四个原本稳定的 Dense target 也出现相同退出竞态，因此决定回退整个 shutdown event/DeepEP reset 实验。

### Round 8：回退 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`FAILED`，1 个 Bazel test、1/1 query；test 285.2 秒

功能链路完整成功：四个 rank 初始化 P2PConnector，Prefill rank 0/1 均为 `sent=4/4, all_cb_received=1`，HTTP 200、actual 与 golden 一致。退出重新呈现分支原有缺陷：Prefill 出现 `PyThreadState_Get` GIL fatal/SIGABRT，Decode 非干净退出。最终结论为“MLA/MoE + CP 字节分片 P2P 功能通过，但 Bazel target 因已由 legacy 对照复现的通用 shutdown 缺陷失败”；最终提交不携带未稳定的 shutdown 实验代码。
