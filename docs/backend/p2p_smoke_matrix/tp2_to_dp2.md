# P2P Smoke：Prefill TP2 → Decode DP2

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_tp2_to_dp2_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct（Dense MHA）
- Prefill：TP2
- Decode：DP2 × TP1
- Query：2 个有公共前缀的连续请求，开启 `reuse_cache`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证 TP head 分片到两个 Decode DP 副本的 route 复制、P2P KV cache 传输和第二个 query 的 cache reuse。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：独立 cache 首次构建

- 日期：2026-08-30
- 源码提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，仅有本场景 target/文档未提交修改
- 命令：

  ```bash
  bazelisk --output_user_root=/home/yanzhan.yzh/.cache/bazel_sm9x_dsv4_p2p_smoke_matrix_cache \
    test --config=sm9x --config=cuda12_9 --nocache_test_results \
    --test_timeout=3600 --run_under=//rtp_llm/test/utils:gpu_lock \
    //rtp_llm/test/smoke:p2p_tp2_to_dp2_decode_entrance
  ```

- 日志：`build_logs/p2p_tp2_to_dp2_round1.log`
- 状态：`INFRA_ERROR`，测试进程未启动。

#### 现象与证据

Bazel 首次展开依赖后并行编译到约 `16058 / 20322`，写 `stats.out` 时返回 `No space left on device`。容器内 `/home/yanzhan.yzh` 所在文件系统为 4.0 TiB，已使用 100%；本轮新 cache 占约 40 GiB。

#### 定位方向

失败发生在 CUDA 编译动作，早于 smoke server 与 query，因此不是 P2PConnector、TransferPlan、输出 golden 或场景配置失败。下一轮删除本轮创建且不完整的独立 cache，并复用此前该分支 smoke 已完成构建的 output base，避免再次展开完整依赖。

#### 修改

无产品源码修改；无测试期望修改。本轮新建的 40 GiB 不完整 cache 已删除，释放约 40 GiB。

### Round 2：复用 warm output base

- 日期：2026-08-30
- 源码提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，仅有本场景 target/文档未提交修改
- 日志：`build_logs/p2p_tp2_to_dp2_round2.log`
- 状态：`INFRA_ERROR`，执行 1 个 Bazel test，smoke server 未启动。

#### 现象与证据

复用此前分支验证的 warm output base 后，Bazel 在 62.402 秒内完成构建并启动 test。runner 导入 `sentence_transformers` 时，Python 又从用户目录加载 `~/.local/lib/python3.10/site-packages/peft` 和 `awq`，最终报错：

```text
ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'
```

#### 定位方向

这是 Bazel runfiles 中的 `transformers` 与容器用户 site-packages 中旧版 `awq` 混用造成的依赖污染。失败早于服务启动和任何 query，不属于 P2P 或测试期望问题。下一轮通过 `--test_env=PYTHONNOUSERSITE=1` 禁止加载用户 site-packages。

#### 修改

无产品源码修改；无测试期望修改。测试命令增加 Python 用户包隔离。

### Round 3：首次完整请求验证

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_dp2_round3.log`
- Bazel 状态：`FAILED`，执行 1 个 test；两个 query 均返回 HTTP 200 并生成 actual。

#### 现象与证据

- Prefill/Decode 各 2 rank，四个 rank 均初始化 `P2PConnector`，TCP backend 生效。
- Prefill 两个发送 rank均报告 `sent=24/24, all_cb_received=1, cancelled=0`，没有 `sendKVCache failed` 或 Decode transfer timeout。
- Query 0 输出与期望一致。
- Query 1 输出文本与期望一致，Prefill reuse 命中 8 token：

  ```text
  reuse_len=8
  prefill_total_reuse_len=8
  prefill_local_reuse_len=8
  ```

- 唯一 diff 是 Decode reuse：期望为 8，实际为 0。

#### 判断与定位方向

这是测试期望需随 DP2 拓扑更新，不是传输实现 bug。两个 query 被 Decode DP 调度到不同副本；Prefill 侧可复用公共前缀，但第二个 Decode 副本没有第一个 query 的本地 cache，因此 `decode_total_reuse_len=0` 合理。响应文本和 P2P 发送完成证据均正常。

#### 修改

- 新增 DP2 场景专属输入 `q_r_dp_sep_p2p_reuse_dp2.json`，保留原 TP-only golden 不变。
- 依据 `outputs.zip/smoke_actual`，仅把第二个 query 的 `decode_total_reuse_len` 和 `decode_local_reuse_len` 改为 0。
- `smoke-golden-fix-from-ci` 脚本已 dry-run；该脚本按设计会忽略简单 response 格式的 `aux_info`，所以两个字段按 actual 证据做最小补丁。

#### 复测计划

使用相同命令和 `--test_env=PYTHONNOUSERSITE=1` 执行 Round 4，确认场景专属期望后 Bazel 通过。

### Round 4：场景专属期望复测

- 日期：2026-08-30
- 日志：`build_logs/p2p_tp2_to_dp2_round4.log`
- 状态：`PASSED`
- 实际执行：1 个 Bazel test，内含 2 个连续 query
- 耗时：Bazel 135.505 秒；test 117.7 秒

#### P2P 与 reuse 证据

- Prefill rank 0：`sent=24/24, all_cb_received=1, cancelled=0`
- Prefill rank 1：`sent=24/24, all_cb_received=1, cancelled=0`
- 未发现 `sendKVCache failed`、`transfers not all done` 或 `P2P cache load failed`
- 两个 `smoke_actual/*.query_0.json`、`query_1.json` 均生成并通过比较
- Query 1：Prefill reuse=8，Decode reuse=0，符合 DP 轮转后的专属期望

#### 结论

TP2 → DP2 的 P2P KV cache 传输有效，两个 Decode DP 副本均接收由两个 Prefill TP rank 分片发送的 cache。该场景在修正 DP 拓扑相关测试期望后通过；未修改产品实现。

### Round 5：最新本地修复上的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_asymmetric_matrix_final_rerun.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；test 99.3 秒

与另外三个矩阵 target 同批、使用 `--nocache_test_results` 实跑。两个 query 均通过，确认 CP position 修复和新增矩阵改动没有回归 TP2→DP2 的 P2P 复制及 DP2 专属 reuse 语义。

### Round 6：shutdown request event 实验暴露退出竞态

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`FAILED`，1 个 Bazel test、2/2 query；test 117.0 秒

两次 query、P2P 发送和 golden 比较均成功；随后 Prefill rank 0 进入 distributed destroy，rank 1 停在 `NormalEngine::stop()`，50 秒后被强杀。失败属于 event 广播 shutdown 实验的跨 rank 竞态，不改变 TP2→DP2 传输功能已通过的结论。

### Round 7：回退 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 315.0 秒

两个 actual 均通过，Prefill rank 0/1 均记录 `sent=24/24, all_cb_received=1`，Decode 两个 rank 初始化 P2PConnector。最终提交采用该结果，并删除 Round 6 的 shutdown 实验源码。
