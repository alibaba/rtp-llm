# BlockTreeCache Benchmark Cases

本文说明 `benchmark_cases.py` 中的 case 家族、选择方式与指标含义；可执行名称和参数以 registry 为准。新增 case 时更新对应家族的实验目的和维度即可，报告组织方式参考 [报告指南](report_template.md)。

## 默认范围与专项选择

- driver 的 `--suite` 和 `--case` 默认均为 `all`；`--suite profile` 未指定 `--case` 时，会运行 registry 中全部 profile case，包括下面的 matrix 和 e2e 家族。
- matrix/e2e 用于专项比较，但当前与基础 case 一同注册在 `PROFILE_CASES` 中，没有独立的专项 suite 或默认排除规则。只想跑某项实验时，用 `--case <完整名称>[,<完整名称>...]` 明确选择；该参数不支持通配符。
- 当前 profile 共 81 个 case：1 个 Tree、16 个 mixed Transfer（含 4 个并发变体）、16 个 batch API matrix 和 48 个 e2e case。家族数量及全局名称唯一性由 `benchmark_driver_profile_test` 锁定，新增 case 时同步更新文档和测试。具体运行范围以本次 `suite_manifest.json` 为准，不依赖文档中的历史数量判定完整性。
- 报告可只分析选定家族或子集，交代选择条件、完成/失败/跳过情况及未展示结果的位置即可，无需为每个注册 case 固定保留一个表格行。

## Smoke（2 个）

| Case | 路径 | 目的 |
| --- | --- | --- |
| `smoke_tree_online_mini` | 在线 lifecycle（runner 内 test-only 小 config，`--task-pool-size=4`） | 最小 match→load-ready→固定 batch sleep→insert→release harness 可执行 |
| `smoke_transfer_d2h_mini` | full_context D2H，c2 / descriptor batch 2，64 requested ops | 最小 Transfer binary/driver/result 路径可执行，并证明 transfer-engine API 实际收到多 descriptor batch |

Smoke 只验证两个预期 case 均生成 completed manifest/result。Tree smoke 走 runner 内部的 test-only 小型 `OnlineTreeWorkloadConfig`（通过环境变量 `BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG=1` 注入，不暴露为公共 CLI 参数），并以 `loads_committed > 0` 证明真实 lower-tier load 路径；Transfer 以 `succeeded_operations > 0` 作为路径哨兵。完整测量有效性与资源 closure 由 native runner 判定，driver/smoke 不再重算内部不变量。

## Profile 家族

### Tree（1 active）

| Case | task pool | 配置 |
| --- | ---: | --- |
| `tree_online_high_variation_c32` | 4 | 固定 online config：C32 逻辑 context、单 scheduler 线程、约 20k 初始节点（3,711 shared base + 16,289 background）、device/host 各 32,768 blocks、20 档长度/13 档 hit-rate、100ms/READY batch forward sleep、15s warmup + 60s measured。混合 BASE/CONTINUATION 请求，family 状态跨 phase 持久化，真实 SWA group-set 语义 |

`tree_online_high_variation_c32` 是**唯一** active/representative Tree perf case（native process timeout metadata = 180s，覆盖 setup + warmup + measured + drain + profiler teardown）。请求形状为 shared base 前缀复用 + 请求独立 key space 的唯一 suffix，混合 append-only continuation（BASE 继承线上长度/hit 分布，CONTINUATION 继承 immediate parent path 并追加唯一 tail）。admission 阶段预分配 load targets 与 suffix blocks，load-before-forward，request refs 跨 forward 持有，deadline 后 drain 全部已 admission request，finalize 校验 active contexts、load tickets、task-pool pending tasks 与 REQUEST refs 零残留。GroupSet fixture 按 profile 类型构造 FULL 或 SWA group set。正式 task-pool 对照由 driver 对同一 case 重复传 `--task-pool-size 4 --task-pool-size 8` 展开，不常驻 registry；任意正整数仍可用于调试。

Tree 的逻辑 block 固定覆盖 256 tokens。20 个请求长度桶（tokens）为 `8000, 14000, 32000, 48000, 92000, 96000, 117000, 120000, 128000, 135000, 141000, 150000, 165000, 200000, 235000, 320000, 480000, 640000, 800000, 950000`，对应权重为 `2, 3, 5, 10, 5, 20, 5, 20, 5, 10, 5, 5, 3, 1, 1, 4, 3, 3, 2, 1`。每个 family 首次请求或抽样长度不大于当前 leaf 时生成 BASE；只有抽样长度更大时才生成 CONTINUATION。BASE 的计划前缀命中率从 `0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99%` 等概率抽样；CONTINUATION 直接继承父 path。正式 native result 把三组分布写入 resolved config，报告据此生成简短的文字摘要。

### Device↔Host mixed（基础 case 与并发变体）

| Case | GroupSet | requested strategy |
| --- | --- | --- |
| `transfer_device_host_full_context_batch` | full_context | batch |
| `transfer_device_host_full_context_staged_sm` | full_context | staged-sm |
| `transfer_device_host_swa_batch` | swa | batch |
| `transfer_device_host_swa_staged_sm` | swa | staged-sm |

以上四个基础 case 都是 `d2h,h2d` mixed window、8-lane wave，每次同方向向 transfer engine 提交最多 8 个 descriptor。device descriptor 按 profile member 展开：默认 DSV4-Flash profile 的 full_context 为 3 pools/62 layer tiles，swa 为 3 pools/85 layer tiles。benchmark-only recorder 记录 measured window 实际命中的 Device↔Host copy strategy；显式策略必须严格命中，fallback 或 mixed 命中都会使 case 失败。这里的 `batch` strategy 指底层 CUDA batch copy，与 transfer-engine descriptor batch 是两个独立层次。

每个基础 case 还有一个 `_c64_b8_w8` 后缀的并发变体：transfer concurrency 为 64、descriptor batch 为 8。后缀只用于识别，实际 worker 数等参数以 registry 和结果中的 resolved config 为准（当前这些变体未单独覆盖 `transfer_worker_count`，使用默认值 1）。报告可把基础配置和并发变体并列比较，并列出实际配置。

### Device↔Disk（4 个）

| Case | GroupSet | mode | working set |
| --- | --- | --- | ---: |
| `transfer_device_disk_full_context_direct` | full_context | direct | auto（c×4） |
| `transfer_device_disk_full_context_buffered` | full_context | buffered | 32768 blocks |
| `transfer_device_disk_swa_direct` | swa | direct | auto（c×4） |
| `transfer_device_disk_swa_buffered` | swa | buffered | 4096 blocks |

方向为 `d2disk,disk2d`。device 仅分配 8 个可复用 lane blocks，disk 分配并寻址完整 working set；每个 wave 先完成全部 write，再提交相同 coordinate 的 read。当前 engine 的 Device→Disk 合约只接受 singleton，runner 通过 case 生命周期内的固定 worker pool 在 wave 内并发提交 8 个 singleton；Disk→Device 则真正提交最多 8 个 descriptor 的 batch，报告按方向展示实际 avg/max batch size。

### Host↔Disk（4 个）

| Case | GroupSet | mode | working set |
| --- | --- | --- | ---: |
| `transfer_host_disk_full_context_direct` | full_context | direct | auto（c×8） |
| `transfer_host_disk_full_context_buffered` | full_context | buffered | 32768 blocks |
| `transfer_host_disk_swa_direct` | swa | direct | auto（c×8） |
| `transfer_host_disk_swa_buffered` | swa | buffered | 4096 blocks |

方向为 `h2disk,disk2h`。两个方向都按最多 8 个 descriptor 的 batch 提交；host 不需要 GPU staging，不限制最小操作数。

### Batch API matrix（单方向）

`_batch_api_matrix_cases()` 动态生成名称：

```text
matrix_{label}_{direction}_{group_set}_{strategy}_c640_t{tasks}_d{descriptors}_batch_api
```

固定 transfer concurrency 为 640；C 为每次提交 1 个 descriptor、对应 640 个 task，D 为每次提交 8 个 descriptor、对应 80 个 task。策略 `cuda_batch`/`staged_sm` × group set `full_context`/`swa` × 方向 `d2h`/`h2d` × C/D 两种提交粒度，共 16 个 case。这里的 task 数是命名中的提交划分，不是 worker 线程数。

例如 `matrix_C_d2h_full_context_cuda_batch_c640_t640_d1_batch_api`。完整名称由循环拼接，源码中搜索 `_batch_api_matrix_cases` 可找到定义。

用于观察 transfer-engine API 提交粒度与底层 copy strategy 的影响。报告可以按方向/group set 组织 C/D 对照表或图，说明 concurrency、实际 descriptor batch 和策略，并区分单方向吞吐与 mixed 吞吐。不要求并入固定的 Device↔Host 表。

### E2E business matrix（单方向）

`_e2e_business_cases()` 动态生成名称：

```text
e2e_B_{direction}_{group_set}_{strategy}_u{upper}_l{lower}_n160_b8
```

方向 `d2h`/`h2d` 各有 `cuda_batch`、`staged_sm` 两种策略；`h2disk`/`disk2h` 各使用 `direct`。每种组合展开 `full_context`/`swa`、上层业务并发 1/4、下层 worker 数 1/4，共 48 个 case。每个业务包含 160 个 descriptor，API descriptor batch 为 8，working set 为 640 blocks。driver 先按 profile payload 与最小字节数上调操作数，再向上对齐到 160 的整数倍，保证完整业务请求；case manifest 的 `resolved_transfer_operation_count` 记录送入 native 的值，native pilot 仍可继续上调，最终测量操作数以 result 为准。

用于观察业务并发和 transfer worker 并发的组合影响。报告可按方向、group set 和策略分别展示 2×2 并发矩阵，或仅展示与实验问题有关的组合。这里的 e2e 是 transfer 业务提交路径，不代表完整模型推理；不要解释为线上请求吞吐。

### 结果呈现与比较范围

报告无需复刻 registry 的结构。可以按介质、方向、策略、提交粒度或业务并发组织，也可以将专项实验单独成篇。分类和标签依据实际参数/传输方向，避免仅从 case 名称推断配置。展示汇总或子集时说明筛选条件，并提供完整 manifest 和原始结果位置，使读者能区分“未运行”“失败/跳过”和“已完成但未展开”。

比较时说明哪些变量相同、哪些不同；例如单方向 matrix 与双向 mixed case 的吞吐口径不同，不能直接当成同负载下的加速比。Tree 的 tp4/tp8 对照还应核对 profile、seed/repetition、trace、capacity、forward sleep、binary SHA 与代码 commit；存在其他差异时披露，避免将差异全部归因于 task pool。

## Profile 结果字段

### Tree lifecycle

| 字段 | 说明 |
| --- | --- |
| `completed_request_transactions` | 完整 match→forward→insert 请求数 |
| `completed_base_transactions` / `completed_continuation_transactions` | 完成的 BASE/CONTINUATION 请求数 |
| `completed_continuation_family_count` | 完成 continuation 的 family 数；正式 workload 必须为 32 |
| `forward_batches` / `forward_requests` | 固定 100ms sleep batch 数及覆盖请求数 |
| `loads_committed` / `loads_succeeded` | lower-tier load 尝试与成功数 |
| `held_request_blocks_peak` | 跨 forward 持有 blocks 峰值 |
| `joined_target_blocks_total` | joined descriptor 通过 match 转移给 request 的真实 target blocks 累计数 |
| `dependency_skip_count` | 因 parent 未就绪而跳过的 CONTINUATION 请求数 |
| `dependency_failed_descendants` | 因 parent 失败而阻断的 descendants 数 |
| `pressure_ready` | warmup 后压力观察值，不是硬 PASS 条件 |
| `load_tickets_pending_peak` | 同时在途 load ticket 峰值，仅作异步行为观察 |
| `final_active_requests` / `final_pending_load_tickets` / `final_pending_tasks` / `final_request_ref_blocks` | finalize 后必须全部为 0 |
| `drain_timeouts` | 各阶段有界 drain 超时次数，必须为 0 |

### Transfer

| 字段 | 说明 |
| --- | --- |
| `mixed_throughput_bps` | 同一 measured window 内双方向成功字节数之和 / 墙钟时间 |
| `direction.<dir>.throughput_bps` | 单方向吞吐 |
| `logical_throughput_bytes_per_second` | 同 `mixed_throughput_bps` |
| `requested_copy_strategy` / `actual_copy_strategy` | 请求与实际使用的传输策略 |
| `requested_transfer_descriptor_batch_size` / `resolved_transfer_descriptor_batch_size` | transfer engine API 请求/解析后的 descriptor batch 大小；与 CUDA copy strategy 无关 |
| `descriptor_batch_size_avg/max` | measured window 内每次 `submit()` 携带的 descriptor 数量；另有 `direction.<dir>.*` 分方向值 |
