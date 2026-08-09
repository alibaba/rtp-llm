# BlockTreeCache Benchmark Cases

本文是 `benchmark_cases.py` 的权威说明。当前 registry 共 16 个 case：2 smoke + 14 profile。

## Smoke（2 个）

| Case | 路径 | 目的 |
| --- | --- | --- |
| `smoke_tree_mini` | scaled Tree，64 nodes，2s warmup + 5s measured | 最小 Tree insert/match/load harness 可执行 |
| `smoke_transfer_d2h_mini` | full_context D2H，c1，64 requested ops | 最小 Transfer binary/driver/result 路径可执行 |

Smoke 只验证预期 case 执行并生成 completed result，不承担 working-set、stateful continuation/fork 或 strategy 语义回归。

## Profile（14 个）

### Tree（2 个）

| Case | threads | 共同配置 |
| --- | ---: | --- |
| `tree_stress_100k` | 8 | flattened/scaled，100k target nodes，fixed 1000-key requests，full/partial/miss=0.7/0.2/0.1 |
| `tree_stress_100k_single` | 1 | 与上项相同，作为单线程对照 |

Tree profile 是 metadata microbenchmark，不包含未实现的 `device-only-block-count` 或 `evict-batch-coordinates` 参数。

### Device↔Host（4 个）

| Case | GroupSet | requested strategy |
| --- | --- | --- |
| `transfer_device_host_full_context_batch` | full_context | batch |
| `transfer_device_host_full_context_staged_sm` | full_context | staged-sm |
| `transfer_device_host_swa_batch` | swa | batch |
| `transfer_device_host_swa_staged_sm` | swa | staged-sm |

四个 case 都是 `d2h,h2d` mixed window、8 workers。device descriptor 按 profile member 展开：full_context 为 3 pools/91 layer tiles，swa 为 3 pools/121 layer tiles。benchmark-only recorder 记录 measured window 实际命中的策略；显式策略必须严格命中，fallback 或 mixed 命中都会使 case 失败。

### Device↔Disk（4 个）

| Case | GroupSet | mode | working set |
| --- | --- | --- | ---: |
| `transfer_device_disk_full_context_direct` | full_context | direct | auto（c×4） |
| `transfer_device_disk_full_context_buffered` | full_context | buffered | 32768 blocks |
| `transfer_device_disk_swa_direct` | swa | direct | auto（c×4） |
| `transfer_device_disk_swa_buffered` | swa | buffered | 4096 blocks |

方向为 `d2disk,disk2d`。device 仅分配 worker slots，disk 分配并寻址完整 working set；相邻 write/read 共享 coordinate。

### Host↔Disk（4 个）

| Case | GroupSet | mode | working set |
| --- | --- | --- | ---: |
| `transfer_host_disk_full_context_direct` | full_context | direct | auto（c×4） |
| `transfer_host_disk_full_context_buffered` | full_context | buffered | 32768 blocks |
| `transfer_host_disk_swa_direct` | swa | direct | auto（c×4） |
| `transfer_host_disk_swa_buffered` | swa | buffered | 4096 blocks |

方向为 `h2disk,disk2h`。host 仅分配 worker slots，disk 是完整 addressable working set。

## 公共判定

- 全部 transfer operation count 都是全局总数。
- mixed throughput 是同一窗口内成功字节的 aggregate throughput。
- measured 至少遍历一轮 working set，且 warmup/pilot/measured cursor 连续。
- 每方向 attempted/succeeded/failed 必须与总计数闭环。
- buffered 每 repetition 独立 drain、采样并清理。
- round-trip case、options、runner 和报告入口均不存在。

代表 perf case：两个 Tree、full_context batch/staged、full_context device-disk direct、full_context host-disk direct，共 6 个。
