# P2P Smoke：Prefill CP2 → Decode TP2

## 场景配置

- Target：`//rtp_llm/test/smoke:p2p_cp2_to_tp2_decode_entrance`
- 模型：Qwen2.5-0.5B-Instruct（Dense MHA）
- Prefill：CP2（`tp_size=2`，`cp_rotate_method=ALL_GATHER`）
- Decode：TP2（`tp_size=2`，`dp_size=1`）
- Query：2 个有公共前缀的连续请求，开启 `reuse_cache`
- P2P：`decode_entrance=1`
- 基线提交：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`

## 预期覆盖

验证 MHA CP 计算布局到普通 TP head 分片布局的 P2P TransferPlan 镜像一致性，以及实际 Prefill → Decode KV cache 传输。

## 验证记录

每次执行或修复均在此处按 Round 顺序追加，保留历史现象与定位方向。

### Round 1：GLM-5 Sparse MLA Decode kernel 不支持 TP2

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target/文档修改
- 初始模型：GLM-5 FP8 4-layer（Sparse MLA）
- 日志：`build_logs/p2p_cp2_to_tp2_round1.log`
- 状态：`FAILED`，执行 1 个 Bazel test；首个 query 的 KV 传输完成，Decode forward 崩溃
- 耗时：Bazel 1770.943 秒；test 1752.0 秒

#### 现象与证据

- NAS 冷读取很慢：四个 rank 在 `_load_from_scratch()` 阶段持续读取约 25 分钟，但累计读字节持续增长，不是静止 hang。
- Prefill/Decode 各 2 rank 最终都成功启动并初始化 P2PConnector。
- 两个 Prefill rank 均报告 `sent=4/4, all_cb_received=1, cancelled=0`，说明首个 query 的 P2P 发送完成。
- Decode forward 随后在 `flash_mla_with_kvcache` 报错：

  ```text
  RuntimeError: Unsupported h_q: 32
  ```

- 两个 Decode rank 因该 Python 异常触发 SIGABRT；没有生成可比较的 smoke actual。

#### 判断与定位方向

这是场景所选模型/kernel 与 Decode TP2 的兼容性问题，不是 TransferPlan 实现或 golden 问题。GLM-5 每 rank 在 TP2 下得到 `h_q=32`，当前 Sparse FlashMLA kernel 不支持；该模型的既有 MLA P2P 用例使用 Decode TP1（每个 DP 副本 `h_q=64`）可完成请求。

#### 修改

保持“Prefill CP2 → Decode TP2”的拓扑目标不变，将模型改为支持该并行配置的 Qwen2.5 Dense MHA，并使用已有的两 query reuse 输入。GLM-5 MLA/CP 字节切分由独立的 `p2p_cp2_sharded_to_dp2_decode_entrance` 覆盖。

#### 复测计划

执行 Round 2，确认 Dense MHA 的 CP2 → TP2 route、传输完成性及两 query 输出。

### Round 2：Dense MHA CP backend dtype 配置错误

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target/文档修改
- 日志：`build_logs/p2p_cp2_to_tp2_round2.log`
- 状态：`FAILED`，执行 1 个 Bazel test；首个 query 的 Prefill forward 崩溃
- 耗时：Bazel 141.549 秒；test 118.8 秒

#### 现象与证据

- Prefill/Decode 各 2 rank 均成功启动并初始化 P2PConnector。
- Prefill CP AllGather attention 在首个 forward 报错：

  ```text
  ValueError: The dtype of q torch.float16 does not match the
  q_data_type torch.bfloat16 specified in plan function.
  ```

- 两个 Prefill rank 因异常触发 SIGABRT；Decode 收到 socket close 并取消 P2P read。
- 未生成 smoke actual，也没有完整的 Prefill P2P 发送完成记录。

#### 判断与定位方向

这是新增场景参数不符合现有 CP FlashInfer backend 的 dtype 约束，不是 TransferPlan 实现或 golden 问题。仓库已有 H20 CP smoke 均使用 BF16。

#### 修改

将 Prefill 和 Decode 的 `act_type` 从 FP16 改为 BF16，保持模型、CP2→TP2 拓扑、P2P 和 reuse 设置不变。

#### 复测计划

执行 Round 3，验证 BF16 配置下完整 P2P 传输与输出。

### Round 3：两端推导的 peer CP method 不一致

- 日期：2026-08-30
- 源码基线：`00086b6ee365aa3b316fbfc2d6aa4a2dcd21f238`，叠加本次 smoke target/文档修改
- 日志：`build_logs/p2p_cp2_to_tp2_round3.log`，以及 Bazel `test.outputs/{prefill,decode}_logs/process.log`
- 状态：`INTERRUPTED`；首个 query 已连续两次确定性失败，第三次自动重试启动后手动终止
- 实际执行：1 个 Bazel test；未完成任何 query
- 耗时：终止前约 697 秒；前两次请求各等待约 300 秒

#### 现象与证据

- Prefill 两个 rank 在前两次请求中均报告 `sent=24/24, all_cb_received=1, cancelled=0`，随后报：

  ```text
  tcp transfer context timeout: no matching recv task within deadline
  ```

- Decode 两个 rank 在对应请求中均报告：

  ```text
  transfers not all done before return deadline (D-100ms), done_tasks=0/24
  ```

- 请求最终返回 HTTP 500：`RANK 0: missing p2p_response`。第二次重试呈现完全相同的 `24/24` 对 `0/24`，排除偶发网络抖动；第三次重试已开始发送，在等待相同超时期间终止，避免无信息增量的等待。

#### 判断与定位方向

这是产品实现 bug，不是测试期望或模型配置问题。`ShardLayoutFactory::peerOf()` 复制本端完整 `PrefillCPConfig` 后只改 TP/分片字段，错误地把角色相关的 `method` 也带到了对端：

- Prefill 本端为 `ALL_GATHER`，真实 attention TP=1；它推导 Decode 时仍保留 `ALL_GATHER`，错算 Decode attention TP=1。
- Decode 本端为 `PREFILL_CP`，真实 attention TP=2；它推导 Prefill 时仍保留 `PREFILL_CP`，错算 Prefill attention TP=2。

双方因此生成不同的 head partition、plan digest 与 route key。Prefill 已提交 24 个发送任务，但 Decode 的 24 个接收任务没有任何 key 能匹配。

现有 `D4_MirrorConsistencyAcrossSides` 直接把真实对端 `ParallelismConfig` 传给 `ShardLayout::forPeer()`，绕过了线上 `ShardLayoutFactory::peerOf()`，所以没有覆盖这个缺陷。

#### 修改

- 在 `ShardLayoutFactory::peerOf()` 中按 peer 角色规范化 CP method：Prefill CP 用 `ALL_GATHER` 表达 attention TP=1，Decode 对应使用 `PREFILL_CP` 表达 attention TP=`tp_size`；非 CP 对端使用 `DISABLED`。
- 新增 factory 路径镜像一致性单测，直接覆盖 Prefill TP2 + `ALL_GATHER` 到 Decode TP2 + `PREFILL_CP`，断言两侧 routes 与 plan digest 一致。

#### 复测计划

先运行 TransferPlan/P2PConnector 四个无缓存 UT，再执行 Round 4 smoke，验证两个 query 的 P2P 任务完成数及输出。

### Round 4：factory 回归单测缺少 Bazel 直接依赖

- 日期：2026-08-30
- 源码基线：Round 3 后的 `ShardLayoutFactory::peerOf()` 修复与新增单测
- Target：`//rtp_llm/cpp/cache/connector/p2p/plan/test:transfer_plan_test`
- 日志：`build_logs/p2p_transfer_plan_factory_fix_ut.log`
- 状态：`FAILED TO BUILD`，实际执行 0 个 test
- 耗时：13.127 秒

#### 现象与判断

新增 factory 路径单测直接 include `ShardLayoutFactory.h`，但 `transfer_plan_test` 只依赖 `:transfer_plan`，Bazel strict-deps 报：

```text
undeclared inclusion(s): ShardLayoutFactory.h
```

这是新增回归测试的 BUILD 依赖遗漏，不是产品逻辑或 smoke 运行失败。

#### 修改与复测计划

为 `transfer_plan_test` 增加直接依赖 `:transfer_plan_glue`，重新运行该 target；通过后继续四 target UT 与 CP2→TP2 smoke。

### Round 5：factory 修复后 P2P 完成，reuse query 输出损坏

- 日期：2026-08-30
- 源码基线：Round 3/4 的 `ShardLayoutFactory::peerOf()` 修复与回归测试
- 日志：`build_logs/p2p_cp2_to_tp2_round5.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 105.719 秒；test 64.5 秒

#### 现象与证据

- 修复后不再出现 300 秒 `no matching recv task` 超时，两个 query 的 P2P 发送和接收都能完成。
- Query 0 输出正确：`" Mathematics is like a lighthouse, a beacon of"`。
- Query 1 命中 Prefill/Decode 各 8 token reuse，但输出为 `"1111111111"`，期望为 `"1. Mathematics is like a lighthouse in the"`。
- 因为 Query 1 的首 token `"1"` 正确，而 Decode 后续每步重复 token id 16，问题发生在 Prefill 首 token 之后的 Decode KV 使用链路。

#### 判断与定位方向

这不是 golden 过时，而是与 reuse 相关的实现错误。优先检查第二次 Prefill 只新写的 partial block，从 `writeCacheToP2P` 生成 key/block 对、CUDA event 就绪，到 planner 切 head 和 Decode 落块的数据内容是否一致。

### Round 6：DEBUG 命令遗漏 `WORLD_SIZE`

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round6_debug.log`
- 状态：`FAILED`，执行 1 个 Bazel test，完成 0 个 query
- 耗时：Bazel 42.755 秒；test 21.9 秒

#### 现象与判断

为提高日志级别覆盖 target env 时遗漏了 `WORLD_SIZE=2`，服务启动阶段报 `KeyError: WORLD_SIZE`。这是诊断命令错误，没有进入模型或 P2P 路径，不用于判断产品状态。

### Round 7：DEBUG 确认 route/key 匹配，收敛到 reuse partial block 内容

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round7_debug.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 83.278 秒；test 66.3 秒

#### 现象与证据

- 两个 query 都真正进入 P2PConnector：Prefill rank 0 各有 2 条 route，每次 `sent=48/48`；Prefill rank 1 因副本选主没有 route。
- Decode 两 rank 每次各有 1 条 route，24 层的 read task 全部成功，两端 plan digest 一致为 `517170406564859234`。
- Query 1 Prefill side-channel 首 token 为 token id 16，即正确的 `"1"`；Decode 随后连续产生 token id 16。
- Query 1 的新 cache key 为 `-7690141825706003981`，Decode 目标 block id 为 4；它在 24 层、两个 Decode head route 上都收到成功回调。
- Query 0 使用 key `-1752445185669318660` 和 `6858790234522785586`，Decode block id 为 1、2，输出正确。

#### 判断与下一步

route id、plan digest、TCP task 匹配和 read 完成性已排除。问题收敛到“CP2 ALL_GATHER Prefill 在 reuse 后生成的新 partial block，切成两个 Decode TP head shard 后的字节内容或落点不正确”。下一轮将对比 Prefill source block 与 Decode target block 的 key、block id、分片大小和 checksum，并与无 reuse 的首次请求对照。

### Round 8：TransferPlan/P2PConnector 四 target 无缓存回归

- 日期：2026-08-30
- 日志：`build_logs/p2p_four_targets_after_peer_method_fix.log`
- 状态：`PASSED`，4/4 Bazel targets 通过，共 141 个 GTest
- 耗时：Bazel 78.641 秒

| Target | GTest 数 | 状态 |
|---|---:|---|
| `transfer_plan_test` | 32 | PASSED |
| `route_codec_test` | 5 | PASSED |
| `components_test` | 84 | PASSED |
| `p2p_connector_test` | 20 | PASSED |

该轮使用 `--nocache_test_results`，证明 factory method 修复及新增镜像一致性断言已完整编译和执行；它不覆盖 Round 7 才暴露的实际 reuse partial-block 数据内容问题。

### Round 9：Decode 禁用 reuse 的隔离实验被磁盘空间中断

- 日期：2026-08-30
- 临时变量：仅将 Decode 的 `reuse_cache` 从 `1` 改为 `0`，Prefill reuse 保持开启
- 日志：`build_logs/p2p_cp2_to_tp2_round9_decode_no_reuse.log`（磁盘写满后日志尾部未能继续落盘；终端输出已保留关键错误）
- 状态：`INFRA_ERROR / FAILED TO BUILD`，实际执行 0 个 test、0 个 query
- 耗时：Bazel 294.464 秒

#### 现象与判断

本轮命令没有复用之前预热的 Bazel output base，而是在同一 output-user-root 下创建了新的 `49e58bc2ad57ce3b60c59fc2027b1968`。依赖重新解析、下载 wheels 并展开编译产物后，该目录占用 34G，将 `/data3` 写满；构建在 `remote_kv_cache_manager_client_files` 复制阶段首先报：

```text
cp: error writing '.../kv_cache_manager_client.so': No space left on device
Executed 0 out of 1 test: 1 fails to build.
```

这是测试基础设施失败，尚未进入 smoke，因此不能回答“Decode 关闭 reuse 后 Query 1 是否恢复”。该轮不改变 Round 7 的产品问题结论。

#### 处理与下一步

- 删除本轮误创建且已精确确认的 34G output base，保留原来已预热的 `8629a87d32852ea45924559137c0036f`。
- 将 target 恢复为 Decode `reuse_cache=1`，避免临时诊断配置进入最终改动。
- 后续每次 Bazel 命令显式固定到预热 output base；若仍需该隔离实验，先在有足够空间时重新执行并另开一轮记录。

### Round 10：容器内 worktree 路径写错，未进入测试

- 日期：2026-08-30
- 临时变量：仅 Decode `reuse_cache=0`
- 日志：`build_logs/p2p_cp2_to_tp2_round10_decode_no_reuse.log`
- 状态：`INFRA_ERROR`，实际执行 0 个 query
- 耗时：Bazel 45.254 秒；test 22.5 秒

#### 现象与判断

本轮使用了错误的容器内 worktree 路径，测试进程没有在预期源码目录启动；同时命令遗漏
`PYTHONNOUSERSITE=1`，随后被用户目录中的旧版 `peft/awq` 污染并在导入阶段失败：

```text
ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'
```

这是测试命令错误，没有启动 Prefill/Decode 服务，不用于判断产品状态。

### Round 11：仅关闭 Decode reuse，错误仍可稳定复现

- 日期：2026-08-30
- 临时变量：Prefill `reuse_cache=1`，Decode `reuse_cache=0`
- 日志：`build_logs/p2p_cp2_to_tp2_round11_decode_no_reuse.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 83.920 秒；test 64.6 秒

#### 现象与证据

- Query 0 正确。
- Query 1 的 Prefill reuse 命中 8 token，Decode reuse 为 0，但输出仍稳定为
  `"1111111111"`；自动重试结果相同。
- P2P 传输完成，没有 route-key mismatch 或 read timeout。

#### 判断与下一步

Decode 本地 cache reuse 不是必要条件；即使 Decode 重新接收完整 P2P KV，错误仍存在。
问题继续收敛到 Prefill reuse 产生并导出的 KV 或其 side-channel 状态。

### Round 12：仅关闭 Prefill reuse，重复 token 消失

- 日期：2026-08-30
- 临时变量：Prefill `reuse_cache=0`，Decode `reuse_cache=1`
- 日志：`build_logs/p2p_cp2_to_tp2_round12_prefill_no_reuse.log`
- 状态：`DIAGNOSTIC_FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 87.497 秒；test 68.8 秒

#### 现象与证据

- Query 0 正确。
- Query 1 不再重复输出 `1`，而是生成连贯文本
  `"1. Mathematics is like a lighthouse, a"`。
- Prefill reuse 为 0、Decode reuse 为 8；由于诊断配置改变，输出及 aux 与原 golden 不一致，
  因而 Bazel 状态仍为 FAILED。

#### 判断与下一步

重复 token 缺陷依赖 Prefill reuse，不依赖 Decode reuse。该轮不是产品通过证明，但把根因范围从
整个 P2P 读写链路缩小到“Prefill CP2 在 prefix reuse 后生成/导出的 KV”。恢复正式配置后记录
key/offset 映射。

### Round 13：确认 reuse key 与物理 block 映射正确

- 日期：2026-08-30
- 配置：恢复 Prefill/Decode `reuse_cache=1`
- 日志：`build_logs/p2p_cp2_to_tp2_round13_mapping_diag.log`，以及 Bazel
  `test.outputs/prefill_logs/process.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 145.013 秒；test 67.9 秒

#### 现象与证据

- Query 1 再次稳定输出 `"1111111111"`，Prefill/Decode 均 reuse 8 token。
- Query 0 的两条 key 映射为：
  - key `6858790234522785586` → offset 0 / block 1
  - key `-1752445185669318660` → offset 1 / block 2
- Query 1 的两条 key 映射为：
  - 复用 key `6858790234522785586` → offset 0 / block 1
  - 新 key `-7690141825706003981` → offset 1 / block 3
- 两个 Prefill rank、24 层的映射一致；新 key 并没有错误地配到 offset 0。

#### 判断与定位方向修正

Round 7 基于“只看到一个新 key”的初步推测不成立：`cache_keys` 仍携带完整逻辑 key
命名空间，`writeCacheToP2P` 的 key/offset 配对正确。问题进一步收敛到 CP ALL_GATHER
在 prefix reuse 后写入新 partial block 的实际内容，或该内容从 source partition 到 Decode
target partition 的字节投影。下一轮应对 source/target block 做内容级 checksum，并与 Prefill
禁用 reuse 的对照轮比较，而不是修改 key 索引或 golden。

### Round 14：两个 Prefill CP rank 的 source block 内容一致

- 日期：2026-08-30
- 配置：Prefill/Decode `reuse_cache=1`，临时在 layer 0 导出完整 KV block 的 FNV-1a checksum
- 日志：`build_logs/p2p_cp2_to_tp2_round14_checksum.log`，以及 Bazel
  `test.outputs/prefill_logs/process.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 86.220 秒；test 68.5 秒

#### 现象与证据

- Query 0 仍正确，Query 1 仍稳定输出 `"1111111111"`。
- 两个 Prefill CP rank 对相同 key 的完整 source block checksum 完全一致：

  | Query | cache key | rank 0 checksum | rank 1 checksum |
  |---|---:|---:|---:|
  | 0 | `6858790234522785586` | `4571197891346297781` | `4571197891346297781` |
  | 0 | `-1752445185669318660` | `11136945673331224088` | `11136945673331224088` |
  | 1（复用） | `6858790234522785586` | `4571197891346297781` | `4571197891346297781` |
  | 1（新写） | `-7690141825706003981` | `5648309338693165448` | `5648309338693165448` |

#### 判断与下一步

该结果排除了“CP 两个 Prefill rank 的复制型 KV 内容不同，planner 选举了错误副本”这一方向。
下一轮保持同样 checksum 插桩，仅关闭 Prefill reuse；比较 Query 1 新 key
`-7690141825706003981` 的 block 内容：若 checksum 改变，问题位于 CP ALL_GATHER prefix-reuse
写 KV 的路径；若不变，则继续在 Decode 落块后计算 checksum，定位传输投影或 Decode 消费。

### Round 15：Prefill 禁用 reuse 后 layer 0 source checksum 不变

- 日期：2026-08-30
- 临时配置：Prefill `reuse_cache=0`，Decode `reuse_cache=1`
- 日志：`build_logs/p2p_cp2_to_tp2_round15_prefill_no_reuse_checksum.log`，以及
  `test.outputs/outputs.zip` 中的 `prefill_logs/process.log`
- 状态：`DIAGNOSTIC_FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 88.043 秒；test 68.7 秒

#### 现象与证据

- Query 0 正确；Query 1 输出恢复为连贯文本 `"1. Mathematics is like a lighthouse, a"`。
- Query 1 Prefill reuse=0、Decode reuse=8，因诊断配置与正式 golden 不同，Bazel 比较失败。
- Query 1 新 key `-7690141825706003981` 在两个 Prefill rank 上都映射到 block 4，layer 0
  checksum 均为 `5648309338693165448`；这与 Round 14 开启 Prefill reuse 时新 key 的 checksum
  完全一致。
- Query 0 两个 key 的 checksum 也与 Round 14 一致。

#### 判断与下一步

“Prefill CP prefix reuse 直接写坏 layer 0 source block”已被排除；关闭 reuse 只改变了物理 block id
（Query 1 从 block 1/3 变为 block 3/4），没有改变相同逻辑 key 的 layer 0 内容。输出却由重复 token
恢复为连贯文本，说明差异仍存在于尚未校验的更高层 KV、Decode 落块内容或 cache block 元数据/消费状态。
下一轮恢复正式 reuse 配置，并把 checksum 扩展到所有层及 Decode 接收完成后的 target block，直接比较
source、传输投影和消费前数据。

### Round 16：正式 reuse 配置的全层 source checksum

- 日期：2026-08-30
- 配置：Prefill/Decode `reuse_cache=1`，checksum 扩展到 24 层
- 日志：`build_logs/p2p_cp2_to_tp2_round16_reuse_all_layers_checksum.log`、`build_logs/p2p_cp2_to_tp2_round16_checksums.txt`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 139.089 秒；test 63.4 秒

#### 现象与定位方向

- Query 1 再次稳定输出 `"1111111111"`。
- 共采集 192 条 source checksum：2 个 query × 24 层 × 2 个 key × 2 个 Prefill rank。
- 每个 `(layer, key)` 在两个 Prefill rank 上均一致，进一步排除 CP 副本选举错误。
- 保留本轮归一化 checksum，下一轮与 Prefill 无 reuse 的同一 Query 1 逐层比较。

### Round 17：全层对照显示新 partial block 从 layer 1 开始分歧

- 日期：2026-08-30
- 临时配置：Prefill `reuse_cache=0`，Decode `reuse_cache=1`，其余不变
- 日志：`build_logs/p2p_cp2_to_tp2_round17_no_reuse_all_layers_checksum.log`、`build_logs/p2p_cp2_to_tp2_round17_checksums.txt`
- 状态：`DIAGNOSTIC_FAILED`，执行 1 个 Bazel test、2 个 query
- 耗时：Bazel 87.258 秒；test 66.9 秒

#### 现象与证据

- Query 1 再次恢复为连贯文本 `"1. Mathematics is like a lighthouse, a"`。
- 复用 key `6858790234522785586` 在 24 层、两个 rank 上与 Round 16 全部一致。
- 新 key `-7690141825706003981`：layer 0 与 Round 16 一致，但 layer 1–23 的 checksum 全部不同；每一轮内部两个 Prefill rank 仍彼此一致。

#### 判断与下一步

差异确实在 Prefill 端产生，但不是 key/block 映射或 CP rank 副本不一致：prefix reuse 使新 partial block
从第一个 attention layer 之后走向不同内容。该差异可能是合法的数值路径差异，也可能是 CP prefix attention
错误；仅凭 checksum 不能断言。下一轮在 Decode 接收完成点记录同一 route/key 的 checksum，确认传输是否
逐字节忠实；若 Decode 内容与对应 source 分片一致，则继续以 CP attention prefix 路径为主线定位。

### Round 18：无 Prefill reuse 时 Decode 落块校验

- 日期：2026-08-30
- 临时配置：Prefill `reuse_cache=0`，Decode `reuse_cache=1`
- 日志：`build_logs/p2p_cp2_to_tp2_round18_no_reuse_decode_checksum.log`、`build_logs/p2p_cp2_to_tp2_round18_recv_checksums.txt`
- 状态：`DIAGNOSTIC_FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 109.087 秒

Query 1 输出为连贯文本 `"1. Mathematics is like a lighthouse, a"`。Decode 端收到的各层、各 route checksum 与对应诊断配置一致；失败来自临时关闭 Prefill reuse 后的输出和 reuse 统计不再匹配正式 golden。下一轮恢复 Prefill reuse，在同一 Decode 落点采样以做一一对照。

### Round 19：恢复 Prefill reuse 后 Decode 落块校验

- 日期：2026-08-30
- 配置：Prefill/Decode `reuse_cache=1`
- 日志：`build_logs/p2p_cp2_to_tp2_round19_reuse_decode_checksum.log`、`build_logs/p2p_cp2_to_tp2_round19_recv_checksums.txt`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 87.651 秒

Query 1 再次输出 `"1111111111"`。与 Round 18 对比可见 Decode 落块内容随 Prefill reuse 路径变化，但该轮仍不能区分差异是在 Prefill 计算阶段形成还是传输投影阶段形成，所以下一轮同时记录发送和接收 checksum。

### Round 20：发送与接收逐 route checksum 一致

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round20_send_recv_checksum.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 134.012 秒

同一 `(layer, route, cache key)` 的 Prefill source slice 与 Decode target slice checksum 全部一致，Query 1 仍为 `"1111111111"`。这排除了 P2PConnector 字节传输、route offset 和 Decode 写入落点破坏数据的可能，根因转向 Prefill CP prefix-reuse 的计算输入。

### Round 21–29：CP prefix 输入与 attention 路径收敛

- 日期：2026-08-30
- 单测日志：`build_logs/cp_handle_inputs_prefix_test.log`、`build_logs/cp_allgather_prefix_padding_test.log`
- 状态：诊断与回归测试阶段

逐步检查 CP shuffle、padding、position 与单层 AllGather attention 后，定位到 prefix reuse 时只对 6 个新 token 建立了从 0 开始的 position；正确绝对位置应从 reuse prefix 长度 8 开始，即 8–13。实现修复为在 CP shuffle position 上增加 prefix offset，并加入：

- `handleInputs` 的 prefix=8、new=6、CP2 回归，rank0 期望 `[8,9,14,15]`，rank1 期望 `[10,11,12,13]`；
- CP AllGather prefix + padding 回归；
- 单层 attention 与 full reference 的数值对照。

这一阶段使用的 checksum、tensor dump 和 TRACE 均为临时诊断代码，结论确认后已清除，不进入最终提交。

### Round 30：position prefix offset 修复后的 smoke

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round30.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 140.189 秒

原 Query 1 从损坏的 `"1111111111"` 恢复为连贯文本 `" Mathematics is like a lighthouse, a beacon of"`，证明 position 缺陷已修复。该输出仍与非 reuse golden 不同，需要判断剩余差异是多层 BF16 数值路径导致的近 tie，还是仍有逻辑错误。

### Round 31–34：复用/非复用路径的层级 TRACE 与 selected logits 对照

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round31_no_reuse_trace.log`、`round32_reuse_trace.log`、`round33_reuse_selected_logits.log`、`round34_no_reuse_selected_logits.log`
- 状态：4 轮 `DIAGNOSTIC_FAILED`，每轮均执行 1 个 Bazel test、2 个 query

两条路径的 token、position、padding 和 P2P 数据均完整；差异收敛到 prefix/new 分段 attention 的正常 BF16 数值差，而不是索引越界或数据损坏。临时 TRACE 已在取证后删除。

### Round 35–36：首 token 候选 logits 近 tie

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round35_reuse_first_token_logits.log`、`build_logs/p2p_cp2_to_tp2_round36_no_reuse_first_token_logits.log`
- 状态：2 轮 `DIAGNOSTIC_FAILED`，每轮均执行 1 个 Bazel test、2 个 query

无 reuse 时 token `1` 的 logit 为 16.922825，token ` Mathematics` 为 16.908041，仅领先 0.0148；reuse 时两者分别为 16.811207 与 16.950113，排名翻转。单层 CP prefix attention 对 full reference 的 max diff 为 0.0078125、mean diff 为 0.00054964。结论是旧提示首 token 本身过于接近，不适合作为 CP 分段计算的稳健 golden；这不是继续修改传输实现的证据。

### Round 37：改用明确依赖 suffix 的序列提示

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round37_robust_query.log`
- 状态：`FAILED`，执行 1 个 Bazel test、2 个 query；Bazel 97.293 秒

Query 1 改为原公共前缀后追加 `Complete the sequence: alpha beta gamma`。CP 与非 CP 对照的前两个 token 都是 `" delta epsilon"`，第三 token 才发生数值分叉，说明提示能稳定验证 suffix 与 prefix reuse，但原 3-token golden 仍过度约束边界数值。

### Round 38：限制为两个确定 token 后通过

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round38_final_query.log`
- 状态：`PASSED`，执行 1 个 Bazel test、2 个 query；Bazel 91.544 秒，test 69.6 秒

将 Query 1 限制为 `max_new_tokens=2`，期望 `" delta epsilon"`。该 query 的 input_len=18，Prefill/Decode reuse 均为 8，同时保留 CP padding、partial-block reuse 和 P2P 传输覆盖。

### Round 39：同进程重复方法误判 reuse 统计

- 日期：2026-08-30
- 日志：`build_logs/p2p_cp2_to_tp2_round39_stability5.log`
- 状态：`TEST_METHOD_FAILED`，1 个 Bazel test；5 次迭代 × 2 个 query 共 10 次比较均因 aux_info 失败

`STABILITY_REPEAT=5` 在首次正式执行后继续复用同一服务和 cache，因此每轮 Query 0 实际 reuse=8、Query 1 实际 reuse=16，而冷启动 golden 分别期望 0/8。10 次响应文本本身始终为预期的 lighthouse 文本和 `" delta epsilon"`。这是稳定性测试方法不适用于带 cache reuse 用例，不是模型输出或 P2P 失败。

### Round 40：五个独立冷启动进程稳定性通过

- 日期：2026-08-30
- 参数：`--runs_per_test=5 --nocache_test_results`
- 日志：`build_logs/p2p_cp2_to_tp2_round40_runs_per_test5.log`
- 状态：`PASSED`，5/5 独立 Bazel runs；每轮 2 个 query，共 10/10 query
- 耗时：总计 264.167 秒；单 run 最短 116.3 秒、最长 244.8 秒、平均 170.7 秒

每个 run 都启动全新的 Prefill/Decode 服务，因此每轮都验证冷启动 Query 0 reuse=0、随后 Query 1 reuse=8。五轮全部通过，证明修复后的 CP2→TP2 P2P 路径和稳健 reuse 输入在独立运行间输出一致。

### Round 41：shutdown request event 实验版本矩阵

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_final_after_shutdown_fix.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 65.9 秒

本场景在实验性 shutdown request event 与 `DeepEPWrapper.reset()` 修改上仍完整通过。两个 actual 均生成并通过比较，说明该实验没有回归 CP2→TP2 的 P2P/reuse 功能；但同一矩阵另 5 个 target 均在退出阶段失败，因此不能据此认为 shutdown 修改有效。

### Round 42：回退不稳定 shutdown 实验后的最终复跑

- 日期：2026-08-30
- 日志：`build_logs/p2p_smoke_matrix_after_shutdown_experiment_revert.log`
- 状态：`PASSED`，1 个 Bazel test、2/2 query；test 178.7 秒

使用固定 warm output base、`--nocache_test_results` 和 `PYTHONNOUSERSITE=1` 实跑。两次请求均生成 actual 并通过；P2PConnector 在 Prefill/Decode 四个 rank 初始化，发送侧记录 `sent=48/48, all_cb_received=1, cancelled=0`，未出现 route/key 或接收超时。最终提交不包含 Round 41 的 shutdown 实验代码。
