# P2P Smoke：Prefill TP2 → Decode DP3

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_tp2_to_dp3_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct（Dense MHA）
- Prefill：TP2
- Decode：DP3 × TP1
- Query：2 个有公共前缀的连续请求，开启 `reuse_cache`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证两个 Prefill TP head 分片向三个 Decode DP 副本展开 route，补充 DP 数大于 2 时的 P2P KV cache 复制与传输完成性覆盖。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：误用 DP2 专属 reuse 期望

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target/文档修改
- 日志：`build_logs/p2p_tp2_to_dp3_round1.log`
- 状态：`FAILED`，执行 1 个 Bazel test；2 个 query 都返回 HTTP 200，第 2 个 query 比较失败
- 耗时：Bazel 254.153 秒；test 233.5 秒

#### 现象与证据

- 2 个 Prefill rank 和 3 个 Decode rank 均成功启动并初始化 P2PConnector，TCP backend 生效。
- Prefill rank 0/1 各报告 `sent=24/24, all_cb_received=1, cancelled=0`。
- 未发现 `sendKVCache failed`、`transfers not all done` 或 `P2P cache load failed`。
- Query 0 输出与期望一致。
- Query 1 输出文本和 Prefill reuse 均一致，唯一 diff 是 Decode reuse：DP2 专属 golden 期望 0，DP3 实际为 8。

#### 判断与定位方向

这是测试期望错误，不是传输实现 bug。DP3 本轮两个连续 query 落到可复用同一 Decode cache 的路径，实际 `decode_total_reuse_len=8`、`decode_local_reuse_len=8`；P2P 发送完成且响应文本完全一致。

#### 修改

将本 target 的输入从 DP2 专属 `q_r_dp_sep_p2p_reuse_dp2.json` 改回仓库已有的 `q_r_dp_sep_p2p_reuse.json`。该文件的第二个 query 正确期望 Decode reuse=8，无需新增重复 golden。

#### 复测计划

保持相同拓扑与无结果缓存参数执行 Round 2，验证期望修正后完整通过。

### Round 2：使用拓扑匹配的 reuse 期望复测

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target/文档修改
- 日志：`build_logs/p2p_tp2_to_dp3_round2.log`
- 状态：`PASSED`
- 实际执行：1 个 Bazel test，内含 2 个连续 query
- 耗时：Bazel 98.253 秒；test 80.7 秒

#### P2P 与 reuse 证据

- Prefill rank 0：`sent=24/24, all_cb_received=1, cancelled=0`
- Prefill rank 1：`sent=24/24, all_cb_received=1, cancelled=0`
- Decode rank 0/1/2 均报告 `P2PConnector initialized without coordinator`。
- 未发现 `sendKVCache failed`、`transfers not all done` 或 `P2P cache load failed`。
- 两个 query 均通过比较；Query 1 Prefill reuse=8、Decode reuse=8。

#### 结论

TP2 → DP3 的 route 展开与实际 P2P KV cache 传输有效。Round 1 仅为误用 DP2 专属 golden，修正测试输入后通过；无产品源码修改。

### Round 3：最终矩阵复跑暴露 Decode DP 命中不确定性

- 日期：2026-08-30
- 日志：`build_logs/p2p_asymmetric_matrix_final_rerun.log`，以及该 target 的 Bazel `test.log`/`outputs.zip`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；test 231.3 秒（其中约 133 秒等待 5 张空闲 GPU）

两个 query 均返回 HTTP 200，响应文本完全正确；Prefill rank 0/1 的发送任务均为 `sent=24/24, all_cb_received=1`。唯一 compare diff 是 Query 1 的 Decode reuse：共用 golden 期望 8，本轮实际 0。与 Round 1/2 中同一 target 曾实际得到 8 对照，说明 DP3 下第二个请求落到哪个 Decode 副本不稳定，固定断言 0 或 8 都会 flaky。

#### 修改与复测计划

新增 DP3 专属输入 `q_r_dp_sep_p2p_reuse_dp3.json`：仍严格断言两次响应、Query 1 的 Prefill reuse=8、输入/输出长度；仅省略由 DP 调度决定的 Query 1 `decode_total_reuse_len` / `decode_local_reuse_len`。这不是接受错误输出，而是移除不属于 P2P 正确性的不确定本地命中计数。Round 4 单独复跑该 target。

### Round 4：DP3 专属稳定期望复测

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_dp3_round4_dp3_golden.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；Bazel 117.472 秒，test 81.2 秒

两个 query 的响应、Prefill reuse 与长度断言全部通过；测试不再把 DP 调度决定的 Query 1 Decode 本地 reuse 数作为稳定契约。TP2→DP3 的 P2P route 和实际 KV 传输回归通过。

### Round 5：shutdown request event 实验暴露退出竞态

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`FAILED`，1 个 Bazel test、2/2 query；test 125.0 秒

两次响应、P2P 与稳定期望均成功；Prefill rank 0 进入 distributed destroy 时 rank 1 尚在 `NormalEngine::stop()`，50 秒后被强杀。该失败与 DP3 golden 无关，属于 event 广播 shutdown 实验。

### Round 6：回退 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 95.0 秒

两个 actual 均通过；Prefill rank 0/1 均记录 `sent=24/24, all_cb_received=1`，Decode 三个 DP rank 均初始化 P2PConnector。最终提交不包含不稳定 shutdown 修改。
