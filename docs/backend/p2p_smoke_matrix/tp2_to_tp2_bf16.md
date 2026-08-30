# P2P TP2 → TP2 BF16 对称对照

## 场景

- Bazel target：`//rtp_llm/test/smoke:p2p_tp2_to_tp2_bf16_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct，Dense MHA
- Prefill：TP2 / DP1 / BF16 / `reuse_cache=1`
- Decode：TP2 / DP1 / BF16 / `reuse_cache=1`
- 链路：`decode_entrance` P2PConnector
- 请求：2 个串行 query，Query 1 复用 Query 0 的 8 token prefix

## 记录约定

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：建立 BF16 非 CP 对照

- 日期：2026-08-30
- 命令参数：`--config=sm9x --config=cuda12_9 --nocache_test_results --test_env=PYTHONNOUSERSITE=1 --run_under=//rtp_llm/test/utils:gpu_lock`
- 日志：`build_logs/p2p_tp2_to_tp2_bf16_round1.log`
- 目的：判断 CP2 → TP2 修复后 Query 1 的输出差异是 CP 实现问题，还是 FP16 期望在 BF16 下过时。
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；Bazel 114.664 秒，test 97.2 秒。

#### 现象与证据

- Query 0：` Mathematics is like a lighthouse, a beacon of`，reuse=0。
- Query 1：`1. Mathematics is like a lighthouse in the`，Prefill/Decode reuse 均为 8。
- Prefill 两个 rank 均初始化 `P2PConnector`；发送侧记录 `sent=24/24, all_cb_received=1`。

#### 判断与定位方向

BF16 本身不会使旧的两-query baseline 分叉；因此 CP2→TP2 的剩余输出差异来自 CP prefix/new 分段数值路径，而不是 dtype 单独变化。

### Round 2：CAT 提示区分度不足

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_tp2_bf16_round2_robust_query_golden.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 141.674 秒

尝试用简单的 CAT 补全提示替代旧近 tie 提示，但实际输出对 suffix 的依赖不够明确，无法可靠证明 reuse 后的新 token KV 被正确消费。该失败属于测试输入设计问题，不修改产品实现，继续寻找更稳健且依赖 suffix 的序列提示。

### Round 3：序列提示的第三 token 仍存在边界分叉

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_tp2_bf16_round3_sequence_query_golden.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 85.537 秒

改为 `Complete the sequence: alpha beta gamma` 后，前两个 token 稳定为 `" delta epsilon"`，但第三 token 在不同 attention 路径仍会分叉。定位为 golden 约束过长，不应把第三个边界 token 当作传输正确性的必要条件。

### Round 4：最终稳健输入通过

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_tp2_bf16_round4_final_query.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；Bazel 81.388 秒，test 64.7 秒

Query 1 最终限制为两个生成 token，期望 `" delta epsilon"`；input_len=18，Prefill/Decode reuse 均为 8。该非 CP BF16 对称基线通过，并与 CP2→TP2 使用完全相同的输入，可用于区分 CP 实现缺陷和通用 P2P/reuse 问题。

### Round 5：最新本地修复上的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_asymmetric_matrix_final_rerun.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；test 94.3 秒

与不对称矩阵 target 同批、使用 `--nocache_test_results` 实跑。两个 query 均通过，继续作为 CP2→TP2 的 BF16 非 CP 对称基线。

### Round 6：shutdown request event 实验暴露退出竞态

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`FAILED`，1 个 Bazel test、2/2 query；test 119.0 秒

两次响应与 golden 均完成，Prefill 两 rank 的 P2P 发送完成；失败只发生在 Decode 关闭阶段。rank 0 已进入 `destroy_distributed_environment()`，rank 1 仍停在 `NormalEngine::stop()`，50 秒后两者被强杀。定位为实验性 event 同时广播后缺少跨 rank 的 engine-loop quiesce 屏障，不是 P2P 或 BF16 输出错误。

### Round 7：回退 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 247.9 秒

回退 event/DeepEP reset 实验后无缓存复跑通过。两个 actual 均匹配；Prefill rank 0/1 分别记录 `sent=24/24, all_cb_received=1`，最终提交保留本场景与 CP/reuse 修复，不保留不稳定 shutdown 修改。
