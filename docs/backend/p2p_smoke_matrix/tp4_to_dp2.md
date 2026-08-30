# P2P Smoke：Prefill TP4 → Decode DP2

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_tp4_to_dp2_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct（Dense MHA，2 KV heads）
- Prefill：TP4
- Decode：DP2 × TP1
- Query：2 个有公共前缀的连续请求，开启 `reuse_cache`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证 Prefill TP 大于 KV head 数时的副本选举、TP4 向两个 Decode DP 副本汇聚，以及实际 KV cache 传输与 reuse。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：模型拓扑合法性检查失败

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本场景 target/文档的未提交修改
- 日志：`build_logs/p2p_tp4_to_dp2_round1.log`
- 状态：`FAILED`，执行 1 个 Bazel test；Prefill 服务启动失败，未发送 query，未进入 P2P
- 耗时：Bazel 192.006 秒；test 171.2 秒

#### 现象与证据

Prefill rank 在模型配置校验阶段报错：

```text
invalid tp_size 4 for config.head_num 14
```

Qwen2.5-0.5B-Instruct 的 attention head 数为 14，不能被 TP4 整除，因此本场景在服务启动阶段即失败。日志中没有 P2PConnector 初始化、TransferPlan、发送完成或请求执行证据。

#### 判断与定位方向

这是场景设计无效，不是 TransferPlan/P2PConnector 实现缺陷，也不是测试期望过时。继续修改产品实现不能使非法模型并行配置成立。

#### 修改

- 从最终 smoke suite 删除 `p2p_tp4_to_dp2_decode_entrance`，保留本文档作为失败验证记录。
- 新增独立的合法替代场景 `p2p_tp2_to_dp3_decode_entrance`：Prefill TP2 → Decode DP3，仍覆盖多源 TP 分片到更多 DP 副本的 route 展开。
