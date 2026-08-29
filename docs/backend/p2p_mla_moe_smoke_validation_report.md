# P2P KV Cache MLA / MoE Smoke 验证报告

## 1. 摘要

- 验证分支：`github/codex/dsv4-p2p-transfer-plan`
- 验证提交：`d43c921f9145b7cc0fb1fc698fe77641bf1b346f`
- 验证日期：2026-08-28
- 环境：CUDA 12.9、SM9x、H20、容器 `yzh`
- 模型 NAS：`9784049a5c-gif85.cn-zhangjiakou.nas.aliyuncs.com:/`，容器内挂载点 `/mnt/nas1`

结论：

1. 原有 MLA/MoE PD smoke 用例没有启用 `decode_entrance`，因此走 legacy cache-store，不覆盖 P2PConnector。
2. 临时增加 `decode_entrance` 版本后，MLA 请求成功完成了真实的 Prefill 到 Decode KV cache 传输；测试最终失败仅发生在服务关闭阶段。
3. MoE 用例真实进入 P2PConnector，但 Prefill 和 Decode 两侧生成了不同的 TransferPlan：Prefill 每层 1 条 route，Decode 每层 2 条 route。plan digest 和传输 key 因而完全不匹配，连续 4 次请求均在 300 秒后失败。
4. MoE 问题是实现缺陷，不是测试期望过时。根因是对端 `ShardLayout` 推导时没有携带对端真实的 `PrefillCPConfig.method` / effective attention TP，错误地使用了本端 CP method。
5. MLA 和 MoE 都暴露了一个独立的服务退出缺陷：关闭阶段调用 Python API 时未持有 GIL，触发 `PyThreadState_Get` fatal。

## 2. 验证范围与用例覆盖

原分支已有以下 PD 用例：

| 用例 | 模型 | `decode_entrance` | 实际传输路径 |
| --- | --- | ---: | --- |
| `mla_cp_pd` | GLM-5 FP8 4-layer | 否 | legacy cache-store |
| `mla_pure_cp_pd` | GLM-5 FP8 4-layer | 否 | legacy cache-store |
| `moe_cp_pd` | Qwen3-30B-A3B-Instruct-2507-FP8 | 否 | legacy cache-store |

原分支已有 Dense `*_decode_entrance` 用例覆盖 P2PConnector，但没有 MLA/MoE 的 P2P smoke。为本次验证临时新增：

- `//rtp_llm/test/smoke:mla_cp_pd_decode_entrance`
- `//rtp_llm/test/smoke:moe_cp_pd_decode_entrance`

两个临时 target 均设置 `enable_decode_entrance=True`，每个 target 包含 1 个逻辑 query，Prefill/Decode 各 2 rank，共使用 4 张 GPU。修改仅存在于本地 `rtp_llm/test/smoke/suites_h20_oss.bzl`，未提交。

## 3. 最终测试结果

| Target | Bazel 状态 | 请求结果 | P2P 结果 | 最终失败点 |
| --- | --- | --- | --- | --- |
| `mla_cp_pd_decode_entrance` | FAILED，2032.0s | query 成功，生成 `query_0.json` | 成功，`sent=4/4, all_cb_received=1` | 服务 shutdown 时 GIL fatal |
| `moe_cp_pd_decode_entrance` | FAILED，1394.8s | 1 个逻辑 query，4 次请求重试均返回 8312 | 失败，Prefill `48/48`，Decode `0/96` | TransferPlan/key 不匹配；随后 shutdown 也有 GIL fatal |

首次将两个 target 串行运行时，MoE 因 NAS 冷加载占用了约 48 分钟，在请求发出后仅剩约 72 秒，最终达到 3000 秒 target timeout。随后使用热页缓存、单独将 MoE 超时提高到 5400 秒重跑，得到上表中的确定性功能失败，因此首次 TIMEOUT 不作为根因结论。

## 4. MLA 现象与判断

### 4.1 P2P 传输证据

MLA 的四个 rank 均初始化了 P2PConnector。请求执行期日志包含：

```text
[PD-DIAG] sendKVCache slow phases ...
total_us=2416977 ... sent=4/4, all_cb_received=1, cancelled=0
```

随后 smoke 生成了：

```text
smoke_actual/.../glm_5_fp8_q_r_h20_cp.query_0.json
```

没有出现 `sendKVCache failed`、`P2P cache load failed` 或接收任务未完成。因此可以确认：

- 请求真实进入 P2PConnector；
- 存在有效的 Prefill 到 Decode KV cache 传输；
- MLA TransferPlan 在该 TP/CP 布局下两侧一致；
- 推理请求本身完成。

### 4.2 Bazel FAILED 原因

请求完成后，Prefill 和 Decode 服务在关闭阶段均出现：

```text
Fatal Python error: PyThreadState_Get:
the function must be called with the GIL held, but the GIL is released
```

`MagaServerManager.stop_server()` 将非干净退出判为失败。因此 MLA 的 Bazel 状态虽然是 FAILED，但这不是 P2P 传输失败或输出错误，而是独立的 Python/C++ shutdown 清理缺陷。

## 5. MoE 现象

### 5.1 可重复性

热缓存重跑中，同一个逻辑 query 共尝试 4 次。四次行为完全一致：

```text
Prefill:
sent=48/48, all_cb_received=1
tcp transfer context timeout: no matching recv task within deadline

Decode:
read: transfers not all done before return deadline
done_tasks=0/96

HTTP:
8312_P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED
RANK 0: missing p2p_response
```

每次请求约 300 秒后失败，与 P2P transfer deadline 对齐。该现象不是随机抖动。

这里的 `all_cb_received=1` 只表示 Prefill 端所有异步发送任务都收到了回调，不表示传输成功；回调携带的是 `no matching recv task` 超时错误。

### 5.2 收发任务数量不一致

模型有 48 层：

- Prefill planner 给实际发送 rank 规划 1 条 route/层，合计 48 个发送任务；
- Decode planner 规划 2 条 route/层，合计 96 个接收任务；
- 96 个接收任务全部未匹配，最终为 `done_tasks=0/96`。

这不是某一层、某一 block 或某个 cache key 缺失，而是所有 route key 均不匹配。

## 6. MoE 根因

### 6.1 两侧真实配置

Prefill：

```text
tp_size: 2
PrefillCPConfig.method: ALL_GATHER
kv_cache_sharded: 0
```

Decode：

```text
tp_size: 1
dp_size: 2
PrefillCPConfig.method: PREFILL_CP
kv_cache_sharded: 0
```

`ParallelismConfig::get_attn_tp_size()` 的规则是：

```cpp
return prefill_cp_config.is_enabled() ? 1 : tp_size;
```

其中 `ALL_GATHER` 的 `is_enabled()` 为真，`PREFILL_CP` 为假。因此：

- Prefill 的真实 effective attention TP 为 1；两个 TP rank 对默认 MHA cache group 是副本，planner 应只选一个源 rank；
- Decode 本端 attention TP 为 1；
- Decode 若要正确推导 Prefill 布局，必须知道对端使用 `ALL_GATHER`，从而得出对端 attention TP 也是 1。

### 6.2 错误的对端布局推导

`ShardLayoutFactory::peerOf()` 当前实现从本端 `ParallelismConfig` 复制一份配置，只替换：

- `tp_size`
- `kv_cache_sharded`
- `prefill_cp_size`
- `role_type`

它没有替换对端真实的 `PrefillCPConfig.method`。

因此 Decode 侧推导 Prefill 时发生以下错误：

```text
Decode 本地 method=PREFILL_CP
        |
        v
peerOf() 把 PREFILL_CP 原样复制到“推导出的 Prefill 配置”
        |
        v
is_enabled() == false，prefill_tp_size=2
        |
        v
错误得到 effective attention TP=2、head_shard_count=2
        |
        v
Decode 生成 2 条 route/层，共 96 个 recv task
```

而 Prefill 使用自己的真实 `ALL_GATHER` 配置：

```text
ALL_GATHER -> is_enabled() == true
           -> effective attention TP=1
           -> 两个 rank 属于同一副本类
           -> 只选 1 条 route/层，共 48 个 send task
```

### 6.3 为什么表现为 key 超时

传输 key 包含：

```text
unique_key + layer_id + cache_tag + route_id + plan_digest
```

两侧 route 数、route 字段及 plan digest 不同，所以 Prefill 发出的 key 在 Decode 的 `TransferTaskStore` 中不存在。TCP 服务一直等待匹配的 recv task，直到 deadline 后返回：

```text
tcp transfer context timeout: no matching recv task within deadline
```

plan digest 被放入 key 的防错机制成功阻止了错误字节写入，但当前只能在 300 秒后以超时暴露配置漂移，失败不够快。

### 6.4 缺陷性质

这是实现 bug，不是测试期望需要更新：

- worker 正确地只执行下发的 route；
- TCP sender/receiver 正确地拒绝了不同 key；
- 错误发生在两侧 scheduler 构造镜像 TransferPlan 的输入不一致；
- 当前“无需协议字段、仅从本端配置推导对端完整布局”的前提不成立，因为 `PrefillCPConfig.method` 是角色相关配置，无法从 `tp_size`、`cp_size` 和 `kv_cache_sharded` 唯一推导。

## 7. 建议修复方案

### 7.1 主修复

不要用本端 `PrefillCPConfig.method` 推导对端 effective attention TP。建议在 peer metadata 中显式携带以下信息之一：

1. 对端 `effective_attn_tp_size`；或
2. 每个 cache group 的 `head_shard_count` / layout fingerprint；或
3. 完整、版本化的 `ShardLayout` 摘要。

首选传递经过归一化的布局事实（effective attention TP 或 per-group head shard count），而不是传递 `CPRotateMethod` 枚举。planner 需要的是物理 KV layout，不应依赖如何产生该 layout 的运行策略。

Prefill 和 Decode 必须使用同一组明确的 source/destination layout 输入生成 TransferPlan。

### 7.2 快速失败

建议在 StartLoad/peer-info 阶段交换或校验完整 plan digest：

- 两侧 digest 不同时立即返回“layout/plan drift”；
- 日志打印两侧 digest、effective TP/CP、tag 和 route 数；
- 不要等待 TCP key 匹配 300 秒后才报超时。

### 7.3 回归测试

至少新增以下测试：

1. Planner 镜像一致性：Prefill `tp=2 + ALL_GATHER`，Decode `tp=1 + PREFILL_CP`，MHA group，`kv_cache_sharded=false`；
2. 断言两侧 plan digest、route 数及每条 route 完全一致；
3. MoE P2P smoke：`moe_cp_pd_decode_entrance`；
4. MLA P2P smoke：`mla_cp_pd_decode_entrance`；
5. 现有 Dense NP1D、ND1P、对称基线，防止修复破坏已通过布局；
6. shutdown GIL 问题单独增加服务启停回归，不与 P2P 正确性混为一项。

## 8. 其他观察

1. 首次冷加载 MoE 时出现的 `RuntimeError: unknown parameter type` 位于 target timeout/终止时刻；热缓存完整重跑未将其识别为首要错误，因此不作为本次根因。
2. MLA 和 MoE 均出现相同的 shutdown GIL fatal，说明它更像通用服务清理问题，而不是特定模型或 P2P route 问题。
3. MoE 模型在本次用例中用于暴露问题，但根因并非 MoE kernel；关键组合是 MHA cache layout 加上两侧不同的 CP rotate method。
4. 当前仅新增了两个本地 smoke target，没有修改产品实现，也没有提交或推送代码。

## 9. 证据与日志

- 首轮 MLA + MoE 日志：
  `/home/yanzhan.yzh/RTP-LLM-dsv4-p2p-test-root/build_logs/p2p_mla_moe_decode_entrance_d43c921.log`
- MoE 热缓存重跑日志：
  `/home/yanzhan.yzh/RTP-LLM-dsv4-p2p-test-root/build_logs/p2p_moe_decode_entrance_d43c921_retry.log`
- MoE Bazel test log：
  `/home/yanzhan.yzh/.cache/bazel_sm9x_dsv4_p2p_transfer_plan_run5_cache/8629a87d32852ea45924559137c0036f/execroot/rtp_llm/bazel-out/k8-opt/testlogs/rtp_llm/test/smoke/moe_cp_pd_decode_entrance/test.log`
- MLA Bazel test log：
  `/home/yanzhan.yzh/.cache/bazel_sm9x_dsv4_p2p_transfer_plan_run5_cache/8629a87d32852ea45924559137c0036f/execroot/rtp_llm/bazel-out/k8-opt/testlogs/rtp_llm/test/smoke/mla_cp_pd_decode_entrance/test.log`

关键源码位置：

- `rtp_llm/cpp/config/ConfigModules.h`：`PrefillCPConfig::is_enabled()`、`ParallelismConfig::get_attn_tp_size()`
- `rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h`：`fromTopology()`、`peerOf()`
- `rtp_llm/cpp/cache/connector/p2p/plan/ShardLayout.h`：`deriveHeadShardCounts()`
- `rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.cc`：Decode 侧 plan 输入与 route 投影
- `rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.cc`：Prefill 侧 plan 输入与 route 投影
- `rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h`：route/digest 传输 key
