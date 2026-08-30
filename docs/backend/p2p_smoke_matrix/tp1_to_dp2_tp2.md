# P2P Smoke：Prefill TP1 → Decode DP2 × TP2

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_tp1_to_dp2_tp2_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct（Dense MHA）
- Prefill：TP1
- Decode：DP2 × TP2（4 ranks）
- Query：2 个有公共前缀的连续请求，开启 `reuse_cache`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证单一 Prefill 源同时向 Decode 的 TP head 分片和两个 DP 副本展开 route，并验证实际 KV cache 传输与 reuse。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：完整验证

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target 与 DP2 专属 golden
- 日志：`build_logs/p2p_tp1_to_dp2_tp2_round1.log`
- 状态：`PASSED`
- 实际执行：1 个 Bazel test，内含 2 个连续 query
- 耗时：Bazel 97.573 秒；test 76.7 秒

#### 现象与证据

- 1 个 Prefill rank 和 4 个 Decode rank 均成功启动。
- Prefill rank 0 报告 `sent=48/48, all_cb_received=1, cancelled=0`。
- 未发现 `sendKVCache failed`、`transfers not all done` 或 `P2P cache load failed`。
- Query 0/1 均生成 actual 并通过比较；Query 1 的 Prefill reuse=8、Decode reuse=0，符合 DP2 轮转语义。

#### 判断与修改

单一 Prefill 源成功向 Decode 的 2 个 DP 副本 × 2 个 TP rank 展开并完成 P2P 传输。未发现实现 bug，无产品源码修改；沿用 Round 3 已建立的 DP2 专属 golden。

### Round 2：最新本地修复上的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_asymmetric_matrix_final_rerun.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；test 166.5 秒

使用 `--nocache_test_results` 实跑。两个 query 均通过，确认单 Prefill 源向 Decode 2 个 DP 副本 × 2 个 TP rank 的 route 展开在最新修复上保持正确。

### Round 3：shutdown request event 实验暴露四 rank 退出竞态

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`FAILED`，1 个 Bazel test、2/2 query；test 117.9 秒

两个 query 与 golden 均成功，Prefill 完成 48 条发送任务。Decode 四个 rank 同时收到 event 后，rank 0/2/3 进入或接近 distributed destroy，rank 1 仍卡在 `NormalEngine::stop()`，50 秒后全部被强杀。定位为 shutdown 实验缺少 collective quiesce，不是四 route 展开或 KV 传输失败。

### Round 4：回退 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 377.6 秒

回退后无缓存复跑完整通过。Prefill rank 0 记录 `sent=48/48, all_cb_received=1`，Decode 四个 rank 均初始化 P2PConnector，两个 actual 均匹配 DP2 专属期望。
