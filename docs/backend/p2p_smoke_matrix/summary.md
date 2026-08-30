# P2P KV Cache 不对称拓扑 Smoke 总测试报告

## 1. 结论

- 验证分支：`origin/codex/dsv4-p2p-transfer-plan`
- 本轮基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238` 加本报告所述未提交修复
- 环境：容器 `yzh`，CUDA 12.9，SM9x，8 × H20，NAS `/mnt/nas1`
- 最终无缓存矩阵：6 个 target 中 5 个 Bazel PASSED、1 个因已知通用 shutdown 缺陷 FAILED；11/11 个实际 query 完成 P2P/输出验证。
- 最终无缓存单测：6/6 targets、171/171 cases PASSED。
- CP2→TP2 另做 5 次独立冷启动稳定性：5/5 runs、10/10 queries PASSED。

TransferPlan 的不对称 TP/CP/DP route、P2P key、CP prefix position 和 partial-block padding 修复已被 Dense MHA 场景与单测覆盖。GLM-5 MLA/MoE + CP 字节分片请求也完成真实 P2P 传输，但服务退出仍会触发既有 GIL/SIGABRT；该问题在不经过 P2PConnector 的 legacy 对照中同样复现，因此不归因于本次传输编排层。

原 MLA/MoE 调查见 [p2p_mla_moe_smoke_validation_report.md](../p2p_mla_moe_smoke_validation_report.md)，已在提交 `00086b6ee` 中单独提交。

## 2. 最终 smoke 矩阵

最终命令使用 `--nocache_test_results`、`PYTHONNOUSERSITE=1`、CUDA 12.9/SM9x、GPU lock 和固定 warm output base。总日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`。

| 场景 | Target | Query | 最终 Bazel 状态 | 功能结论 |
|---|---|---:|---|---|
| Prefill CP2 → Decode TP2 | `p2p_cp2_to_tp2_decode_entrance` | 2/2 | PASSED，178.7s | P2P + reuse 通过 |
| TP2 → TP2 BF16 对称基线 | `p2p_tp2_to_tp2_bf16_decode_entrance` | 2/2 | PASSED，247.9s | P2P + reuse 通过 |
| Prefill TP2 → Decode DP2 | `p2p_tp2_to_dp2_decode_entrance` | 2/2 | PASSED，315.0s | TP 分片向 2 个 DP 副本复制通过 |
| Prefill TP1 → Decode DP2 × TP2 | `p2p_tp1_to_dp2_tp2_decode_entrance` | 2/2 | PASSED，377.6s | 单源向 4 个 Decode rank 展开通过 |
| Prefill TP2 → Decode DP3 | `p2p_tp2_to_dp3_decode_entrance` | 2/2 | PASSED，95.0s | TP 分片向 3 个 DP 副本复制通过 |
| Prefill CP2(sharded) → Decode DP2 | `p2p_cp2_sharded_to_dp2_decode_entrance` | 1/1 | FAILED，285.2s | P2P/golden 通过；shutdown GIL/SIGABRT 失败 |

独立场景报告：

- [CP2→TP2](cp2_to_tp2.md)
- [TP2→TP2 BF16](tp2_to_tp2_bf16.md)
- [TP2→DP2](tp2_to_dp2.md)
- [TP1→DP2×TP2](tp1_to_dp2_tp2.md)
- [TP2→DP3](tp2_to_dp3.md)
- [CP2(sharded)→DP2](cp2_sharded_to_dp2.md)
- [TP4→DP2 无效设计记录](tp4_to_dp2.md)

TP4→DP2 没有纳入最终 suite：Qwen2.5-0.5B 的 `head_num=14` 不能被 TP4 整除，服务在 P2P 初始化前即拒绝配置。它属于场景设计无效，不是实现失败；合法替代覆盖为 TP2→DP3。

## 3. P2P 路径有效性证据

最终六个 target 都设置 `enable_decode_entrance=True`，Prefill/Decode rank 日志均出现 `P2PConnector initialized without coordinator`。Dense 场景的发送完成记录分别覆盖每层 24 或 48 个任务；CP2(sharded)→DP2 的两个 Prefill CP rank 均记录：

```text
sent=4/4, all_cb_received=1, cancelled=0
```

所有 11 个 query 均生成 `smoke_actual/*.query_N.json`；未出现 `no matching recv task`、`transfers not all done` 或 `P2P cache load failed`。因此最终矩阵确实覆盖有效 Prefill→Decode KV cache 传输，而不是只验证服务启动。

## 4. 最终单测

日志：`build_logs/p2p_final_submission_unit_tests.log`。所有 target 使用 `--nocache_test_results` 实际执行。

| Target | Case 数 | 状态 |
|---|---:|---|
| `context_parallel_py_wrapper_test` | 12 | PASSED |
| `test_allgather_cp_impl` | 18 | PASSED |
| `transfer_plan_test` | 32 | PASSED |
| `route_codec_test` | 5 | PASSED |
| `components_test` | 84 | PASSED |
| `p2p_connector_test` | 20 | PASSED |
| 合计 | 171 | PASSED |

## 5. 修复与分类

### 5.1 产品实现 bug

1. `ShardLayoutFactory::peerOf()` 原样复制本端的 CP method，导致两侧对 peer effective attention TP 的推导不同，进而生成不同 route 和 plan digest。修复为按 peer 角色规范化 CP method，并新增 factory 真实路径的 plan 镜像一致性测试。
2. CP prefix reuse 时，`prefill_shuffle_indices` 仍从 0 开始，遗漏 `prefix_length` 偏移；第二个 query 的新 token 会使用错误 position，表现为首 token 正确、后续 Decode token 重复。修复为将局部 zigzag index 转成绝对 position。
3. CP partial block padding 的测试构造把 padding 当作真实 token，未覆盖 6 个新 token pad 到 8 的生产路径。测试工具现分别维护 real/padded length、padding mask 与 paged-cache 期望，并增加两个 CP rank 回归。

### 5.2 测试期望随拓扑更新

1. DP2 下两个连续 query 可能调度到不同 Decode 副本，第二个 query 的 Decode local reuse 合法为 0；新增 DP2 专属期望，不修改响应文本。
2. DP3 的第二个 query 落在哪个 Decode 副本不稳定，Decode local reuse 可为 0 或 8；DP3 专属期望仍严格校验响应、Prefill reuse、输入/输出长度，只移除不属于 P2P 正确性契约的 Decode 本地命中计数。
3. CP2→TP2 使用明确依赖 suffix 的两-token序列补全，避免原长输出的近 tie token 把数值边界差异误判为传输错误。

### 5.3 场景/基础设施问题

- TP4→DP2：模型 head 数与 TP4 不可整除，场景已从 suite 删除并保留独立报告。
- 曾出现独立 Bazel output base 写满磁盘、用户 site-packages 污染和容器路径错误；最终验证固定到 warm output base，并设置 `PYTHONNOUSERSITE=1`，这些轮次不用于产品结论。

### 5.4 shutdown 修复实验的处理

尝试过 multiprocessing shutdown request event、提前 `DeepEPWrapper.reset()` 和 leader/follower event 广播。单次曾通过，但最终矩阵只有 1/6 target 通过：不同场景随机出现某 rank 停在 `NormalEngine::stop()`、另一 rank 已进入 `destroy_distributed_environment()`，50 秒后被强杀。

回退该实验后 Dense 矩阵恢复为 5/5 PASSED。说明实验改变了竞态窗口但没有建立跨 rank 的 collective quiesce 屏障；最终提交已删除全部 shutdown 实验源码，只在场景文档中保留每轮证据。

## 6. 原 MLA/MoE 调查与本轮关系

- 原有 `mla_cp_pd`、`mla_pure_cp_pd`、`moe_cp_pd` 都走 legacy cache-store，不覆盖 P2PConnector。
- 临时 MLA P2P 用例完成 4/4 发送和推理，最终只在 shutdown 失败。
- 原临时 MoE P2P 用例曾因两侧 CP method 推导不同表现为 Prefill 48 个发送任务、Decode 96 个接收任务且 0 个匹配；本轮 `ShardLayoutFactory::peerOf()` 修复正针对该根因，并由 32-case planner 测试及 CP2→TP2 实际 smoke 覆盖。
- 大型 Qwen3 MoE 临时 target 未在本轮最终矩阵重新执行；不能把 factory 单测和 GLM-5 场景替代为该特定 target 的 PASSED 结论。

## 7. 尚存风险与后续方向

1. GLM-5 MLA/MoE 多 rank shutdown 仍未修复。下一步应在挂住窗口实时 attach 所有 rank，确认 `NormalEngine::loop_thread_->join()`、DeepEP persistent kernel 和 `destroy_process_group()` 的相互等待，并在 engine 全体停止与 process-group 销毁之间建立显式屏障。
2. `LayerCacheBufferUtil::buildKeyBlockInfosSliced()` 在切分前置条件不满足时使用 `continue`，可能把布局错误延迟为 `cache_key missing in request`；本轮合法 CP2(sharded) 配置未触发，但建议后续改为快速失败并补异常输入测试。
3. plan digest 不一致目前主要依赖传输 key 超时暴露；建议在 scheduler handshake 阶段交换/校验 digest，避免等待完整 transfer deadline。
4. DP2/DP3 的本地 reuse 指标受调度影响，不应作为跨运行固定 golden；响应、Prefill reuse、P2P 完成数仍需严格断言。
