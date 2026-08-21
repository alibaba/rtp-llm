# BlockTreeCache Benchmark Cases

本文是 `benchmark_cases.py` 的权威说明。当前 registry 共 15 个 case：2 smoke + 13 profile（1 个 active Tree + 12 个 Transfer）。

## Smoke（2 个）

| Case | 路径 | 目的 |
| --- | --- | --- |
| `smoke_tree_online_mini` | 在线 lifecycle（runner 内 test-only 小 config，`--task-pool-size=4`） | 最小 match→load-ready→固定 batch sleep→insert→release harness 可执行 |
| `smoke_transfer_d2h_mini` | full_context D2H，c2 / descriptor batch 2，64 requested ops | 最小 Transfer binary/driver/result 路径可执行，并证明 transfer-engine API 实际收到多 descriptor batch |

Smoke 只验证两个预期 case 均生成 completed manifest/result。Tree smoke 走 runner 内部的 test-only 小型 `OnlineTreeWorkloadConfig`（通过环境变量 `BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG=1` 注入，不暴露为公共 CLI 参数），并以 `loads_committed > 0` 证明真实 lower-tier load 路径；Transfer 以 `succeeded_operations > 0` 作为路径哨兵。完整测量有效性与资源 closure 由 native runner 判定，driver/smoke 不再重算内部不变量。

## Profile（13 个）

### Tree（1 active）

| Case | task pool | 配置 |
| --- | ---: | --- |
| `tree_online_high_variation_c32` | 4 | 固定 online config：C32 逻辑 context、单 scheduler 线程、约 20k 初始节点（3,711 shared base + 16,289 background）、device/host 各 32,768 blocks、20 档长度/13 档 hit-rate、100ms/READY batch forward sleep、15s warmup + 60s measured。混合 BASE/CONTINUATION 请求，family 状态跨 phase 持久化，真实 SWA group-set 语义 |

`tree_online_high_variation_c32` 是**唯一** active/representative Tree perf case（native process timeout metadata = 180s，覆盖 setup + warmup + measured + drain + profiler teardown）。请求形状为 shared base 前缀复用 + 请求独立 key space 的唯一 suffix，混合 append-only continuation（BASE 继承线上长度/hit 分布，CONTINUATION 继承 immediate parent path 并追加唯一 tail）。admission 阶段预分配 load targets 与 suffix blocks，load-before-forward，request refs 跨 forward 持有，deadline 后 drain 全部已 admission request，finalize 校验 active contexts、load tickets、task-pool pending tasks 与 REQUEST refs 零残留。GroupSet fixture 按 profile 类型构造 FULL 或 SWA group set。正式 task-pool 对照由 driver 对同一 case 重复传 `--task-pool-size 4 --task-pool-size 8` 展开，不常驻 registry；任意正整数仍可用于调试。

Tree 的逻辑 block 固定覆盖 256 tokens。20 个请求长度桶（tokens）为 `8000, 14000, 32000, 48000, 92000, 96000, 117000, 120000, 128000, 135000, 141000, 150000, 165000, 200000, 235000, 320000, 480000, 640000, 800000, 950000`，对应权重为 `2, 3, 5, 10, 5, 20, 5, 20, 5, 10, 5, 5, 3, 1, 1, 4, 3, 3, 2, 1`。每个 family 首次请求或抽样长度不大于当前 leaf 时生成 BASE；只有抽样长度更大时才生成 CONTINUATION。BASE 的计划前缀命中率从 `0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99%` 等概率抽样；CONTINUATION 直接继承父 path。正式 native result 把三组分布写入 resolved config，报告据此生成简短的文字摘要。

### Device↔Host（4 个）

| Case | GroupSet | requested strategy |
| --- | --- | --- |
| `transfer_device_host_full_context_batch` | full_context | batch |
| `transfer_device_host_full_context_staged_sm` | full_context | staged-sm |
| `transfer_device_host_swa_batch` | swa | batch |
| `transfer_device_host_swa_staged_sm` | swa | staged-sm |

四个 case 都是 `d2h,h2d` mixed window、8-lane wave，每次同方向向 transfer engine 提交最多 8 个 descriptor。device descriptor 按 profile member 展开：full_context 为 3 pools/91 layer tiles，swa 为 3 pools/121 layer tiles。benchmark-only recorder 记录 measured window 实际命中的 Device↔Host copy strategy；显式策略必须严格命中，fallback 或 mixed 命中都会使 case 失败。这里的 `batch` strategy 指底层 CUDA batch copy，与 transfer-engine descriptor batch 是两个独立层次。

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

## Profile 结果字段

### Tree lifecycle

| 字段 | 说明 |
| --- | --- |
| `completed_request_transactions` | 完整 match→forward→insert 请求数 |
| `completed_base_requests` / `completed_continuation_requests` | 完成的 BASE/CONTINUATION 请求数 |
| `continuation_families_completed` | 完成 continuation 的 family 数；正式 workload 必须为 32 |
| `forward_batches` / `forward_requests` | 固定 100ms sleep batch 数及覆盖请求数 |
| `loads_committed` / `loads_succeeded` | lower-tier load 尝试与成功数 |
| `held_request_blocks_peak` | 跨 forward 持有 blocks 峰值 |
| `joined_holder_blocks_total` | joined descriptor 的真实 target blocks 获得独立 REQUEST holder 的累计数 |
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
