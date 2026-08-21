# BlockTreeCache Benchmark

本目录提供 BlockTreeCache 的 Tree 在线生命周期 microbenchmark 与 Device/Host/Disk transfer benchmark。当前实现以 workload 可核对、失败可传播、repetition 相互独立为前提；整改前的 buffered 大工作集与 round-trip 数值不能作为基线。

> [!IMPORTANT]
> 运行本 benchmark 前必须先阅读并遵循本 README。不建议参考或套用 benchmark 目录外的通用 skill、测试流程或构建参数；除非本文明确引用，构建配置、运行方式、case 定义和结果判定均以本目录文档为准。外部说明与本文冲突时，以本文为准。

## 设计边界

- C++ binary 只有 `tree`、`transfer` 两个子命令；suite/case registry 由 `benchmark_cases.py` 统一维护。
- Tree case 是 scaled/flattened 在线生命周期 microbenchmark：**C32 是 32 个逻辑 request context（内存数据对象，不是线程）**，只有一个 foreground scheduler 线程串行调用 match/insert/release；BlockTreeCache 的共享 load/evict/store task pool 是唯一的 Tree 调节参数（`--task-pool-size`，默认 4）。GroupSet fixture 按 profile 构造真实 group set 类型：FULL 组构造 `FullGroupSet`，SWA 组构造 `SWAGroupSet`（含 profile 的滑动窗口语义）。payload 为 scaled，不复刻 profile 的 member fan-out、device-only group 或真实 GPU forward。
- `tree_online_high_variation_c32` 是**唯一** active/representative Tree perf case。该 case 使用混合 BASE/CONTINUATION 请求：BASE 使用 shared-base + unique suffix（原 high-variation 形状），CONTINUATION 完整继承同一 family 的 parent path 并追加唯一 tail（append-only）。family 状态跨 warmup/measured 持久化，scheduler 正确处理 dependency skip 与 token-budget FIFO。
- Transfer 的 device 侧按 profile member/layer 构造真实 copy tile；device/host lane blocks 只按 wave width 分配，host/disk working set 独立寻址，不按 working set 大小扩张 GPU staging 内存。每个 wave 按方向将独立 endpoint 组成 transfer-engine descriptor batch，再等待该方向完成后推进配对方向。device↔disk staging lease 数至少取 wave width，避免 lane 超订导致 `staging pool exhausted`。
- mixed case 在同一 measured window 内交替执行两个方向。展示的“混合总吞吐”是该窗口内两个方向成功字节数之和除以墙钟时间，不是两个单方向峰值的相加。
- tier round-trip 已删除。demote/load 生命周期正确性继续由现有功能测试看护。
- 计时边界：阶段为 `bootstrap → profile load → allocation → setup → warmup → measured → finalize`，只有 `measured`（含 deadline 后的已 admission request drain）计入指标；perf 运行是独立进程，其 wall time 不进入统计。
- 内存模型：常驻显存由“同时存活的 blocks × 单 block payload”决定，而非 Tree 节点总数。Tree 使用固定 `scaled` payload，device/host pool 各 32,768 blocks；资源预检（preflight）不足时直接终止，不自动缩小命名 case。

## Case 矩阵

- smoke：2 个，只运行最小在线 Tree lifecycle 与 D2H Transfer 端到端路径。
- profile：active Tree 1 个（`tree_online_high_variation_c32`，task pool 4）+ Transfer 12 个（4 个 Device↔Host copy-strategy 对照、4 个 Device↔Disk、4 个 Host↔Disk）。
- representative perf case 共 5 个：1 个在线 Tree + 4 个 Transfer（host CUDA-batch strategy、host staged-sm、device-disk direct、host-disk direct）。
- buffered profile working set：`full_context=32768` blocks，`swa=4096` blocks。每次 measured window 至少完整访问一轮 addressable working set。
- task-pool 对照（tp4/tp8）不是常驻 case：driver 对同一 case 重复传 `--task-pool-size 4 --task-pool-size 8` 展开专项矩阵。每个 repetition 的两组除 pool size 外 profile、seed、repetition identity、workload definition/trace hash、capacity 与固定 100ms forward sleep 必须一致。tp32 不在正式对照矩阵内，但 runner/driver 保留任意正整数参数能力。

权威清单见 [docs/benchmark_cases.md](docs/benchmark_cases.md)。

需要采集 CUDA API/GPU timeline、Copy Engine/PCIe/BAR1 等 GPU Metrics 或导出 `.nsys-rep`/SQLite 时，见 [Nsight Systems 采集指南](docs/nsys_collection_guide.md)。

## 构建

```bash
cd github-opensource

# 同一份普通构建同时用于指标和 CPU perf；DWARF unwind 不改变 C++ action key
bazelisk build -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_gpu_benchmark \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_driver
```

产物在 `bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/`。

只修改 Python case、driver 或报告时，可构建 source-only 的 `:benchmark_driver_sources`，不会分析或准备 GPU binary 的庞大 runfiles；Tree/Transfer runner 与公共 fixture 也已拆成独立 C++ library，缩小 benchmark 自身的增量重编面。

**常见坑**：
- base 配置已经使用 `-O2 -g --strip=never`；driver 通过 `perf --call-graph dwarf,16384` 展开栈，不再维护会使传递 C++ action 全部失效的 frame-pointer perf 配置。
- 修改 C++ runner 后必须重新构建 gpu_benchmark 才能生效；修改 `benchmark_cases.py` 后需重新构建 driver（py_binary 从 runfiles 加载 registry），否则新 case 参数不生效。
- 构建缓存：本仓库只配置了 `fetch:downloader --remote_cache`（拉取外部依赖用），**没有 build 级远程/磁盘缓存**。rebase/改代码后波及的 CUDA 模板库（flashinfer/xqa/cutlass）会整段重编，单文件可达数分钟，属正常现象。

## 运行

### 前置准备（必做）

benchmark binary 依赖 libpython3.10（torch），直接运行会报 `libpython3.10.so.1.0: cannot open shared object file`。设置 `LD_LIBRARY_PATH` 即可（bazel test 的 smoke 目标会自动设置，手动跑 driver 不会）：

```bash
export LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH
```

若机器上 libpython3.10 不在 `/opt/conda310/lib`，用下面命令定位后补 soname：

```bash
find ~/.cache/bazel -name "libpython3.10.so" -path "*/block_tree_cache*" 2>/dev/null | head -1
# 或直接指定任意包含 libpython3.10.so.1.0 的目录
```

### 测试与 smoke

```bash
# 纯逻辑最小看护（无需 GPU）
bazelisk test -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:transfer_benchmark_workload_test \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:tree_workload_generator_test

# 真实 binary + driver 端到端看护（需 GPU）
bazelisk test -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_smoke_test
```

最小看护矩阵只保留三个 target：两个纯逻辑测试分别验证 Tree trace 和 Transfer 调度的基础不变量；GPU smoke 覆盖 binary/driver 启动、case registry、profile 加载以及最小在线 Tree lifecycle 与 D2H Transfer 的真实端到端路径。

### 用 driver 跑 suite（推荐）

```bash
export LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH

# 正式 profile；默认每 case 只运行 1 次，约 25-35 分钟
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --process-repetitions 1 \
  --output-dir /tmp/btc_profile \
  --disk-root /path/to/benchmark_disk

# profile + perf 收集（与普通指标共用同一 binary；自动发现 ~/FlameGraph-master）
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --process-repetitions 1 \
  --output-dir /tmp/btc_profile \
  --disk-root /path/to/benchmark_disk \
  --perf record

# 固定采集 tree_online_high_variation_c32 CPU perf：1 次有效性 repetition + 1 次 profiling process
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --case tree_online_high_variation_c32 \
  --process-repetitions 1 \
  --task-pool-size 4 \
  --task-pool-size 8 \
  --perf record \
  --flamegraph-tools-dir /path/to/FlameGraph \
  --output-dir /tmp/btc_tree_profile
```

在线 Tree case 的 native process timeout 来自 case metadata（180s，覆盖 setup + 15s warmup + 60s measured + drain + profiler teardown），driver 不再有含义歧义的顶层 `--min-measured-seconds`；Transfer binary 自己的 duration option 不受影响。

CPU perf 的宿主/容器执行边界，以及 off-CPU 的 fallback/skip 规则见后文能力矩阵。

默认是严格 suite：任一 failed、partial、skipped 或 required perf 失败都会返回非零。仅调试时可显式传 `--allow-incomplete`。缺少 `--disk-root` 时 disk case 仍保留在 manifest，并记为 `skipped_no_disk`。

### 正式 profile 的 profiling 完整性（必做）

- **正式 profile 默认必须采集 off-CPU**：完成 canonical suite 和 driver 的 on-CPU perf 后，还必须按本文后续固定流程，对 `tree_online_high_variation_c32` 启动独立进程并采集 15 秒 BCC off-CPU。driver 当前不负责自动编排这一步；driver 返回 0 不代表正式报告已经具备完整 profiling 产物。
- 只有 profiler sidecar、host PID/cgroup namespace、内核 BTF/tracefs/debugfs/bpffs、BCC smoke、权限或符号质量等环境预检不满足时，才允许放弃 off-CPU。必须在报告中记录具体失败项，写成 `off-CPU: skipped (<preflight failure>)`；**“用户未单独请求”或“只要求 benchmark”不是合法 skip 原因**。
- 环境预检通过后，缺少有效的 `offcpu.folded`、`offcpu_flamegraph.svg` 或 `offcpu_manifest.json` 时，不得把报告标记为 profiling 完整或直接发布。采集后的 benchmark 生命周期与 missed-stack 质量门禁仍必须全部通过。
- 专项调试可以显式省略 off-CPU，但必须把报告标记为“专项/不完整”，不能冒充正式 profile 报告。

### 直接跑 binary（调试用）

```bash
LD_LIBRARY_PATH=/opt/conda310/lib ./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_gpu_benchmark tree \
  --cuda-device 0 \
  --max-device-memory-fraction 0.8 \
  --model-profile rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/deepseek_v4_pro_fp8_tp1_cp1.json \
  --seed 42 \
  --repetition-id 0 \
  --task-pool-size 4 \
  --output-json /tmp/tree_result.json
```

Tree 的所有 workload 常量（C32、20k 初始节点、32,768 block pools、20 档长度/13 档 hit-rate 分布、100ms forward sleep、15s warmup、60s measured）都是固定值并写入 resolved config，不从 CLI 修改；`--task-pool-size` 是唯一 Tree 参数。smoke 使用 runner 内部 test-only 小型 config（环境变量 `BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG=1` 注入），不暴露为公共参数。

> 注意：binary 的参数名与 driver 的 case 参数名**不同**。driver 会做翻译（例如 driver 的 `--disk-root` 对每个 repetition 展开成 binary 的 `--disk-path <rep 子目录>`）；手动跑 binary 时不要照抄 case 参数，直接看 `--help`。

## Transfer 语义与结果核对

`--transfer-operation-count` 是 measured window 的全局总操作数，不是每 lane 或每 batch 的操作数。pilot 只允许向上扩大：

```text
final_operations = max(requested, calibrated, working_set_blocks * direction_count)
```

logical coordinates 被切成不超过 `--transfer-concurrency` 的 wave。同一 wave 复用一组独立 endpoint lanes，并按方向建立 descriptor batch；一个方向完成后才推进下一个方向，因此 paired write/read 共享 working-set index 且保持依赖顺序：

```text
lane_index = coordinate_offset_in_wave
logical_coordinate = operation_index / direction_count
working_set_index = permutation(logical_coordinate) % working_set_blocks
```

`--transfer-descriptor-batch-size` 控制一次 transfer-engine `submit()` 最多携带多少 descriptor，`0` 表示跟随 concurrency；正式 case 显式设为 8。它与 `--copy-strategy=batch` 不同：前者控制 transfer-engine API 的 descriptor 组批，后者控制 Device↔Host copy plan 是否使用 CUDA batch copy。当前 Device→Disk 引擎合约仅接受 singleton，runner 在 wave 内并发提交 singleton 以保留 lane concurrency；其余五个方向走真实 descriptor batch。

正式结果必须满足：

```text
requested = attempted = succeeded + failed
failed = 0
requested working set = addressable working set = visited working set
```

result 还记录每个 tier 的 capacity/allocated/addressable blocks、每方向 attempted/succeeded/failed、首个错误、descriptor batch 的 requested/resolved/actual avg/max、requested/actual copy strategy 和是否 wrap-around。显式 `batch`/`staged-sm` 会在 benchmark 配置中关闭另一个 Device↔Host copy 优化路径，并由 benchmark-only recorder 记录 measured window 实际命中的策略；任何 fallback 或 mixed 命中都会使 case 无效。

所有 host/device block 在运行前清零。mixed disk case 的 read 跟随同 coordinate 的 write；如果方向顺序无法提供该前置条件，setup 会预填完整 disk working set。warmup、pilot、measured 使用连续 coordinate cursor，不从 offset 0 重放。

## Tree 语义（在线生命周期）

- 固定 workload：setup 构造约 20k 节点（3,711-block shared base + 16,289-block background tree），device/host pool 各 32,768 blocks，watermark 0.8/0.9。shared base 不 pin，持续唯一 suffix insert 会驱动 demote/load/evict churn。
- 请求 path = shared base 前缀（按 13 档 hit-rate 采样的 planned reuse）+ 请求独立 key space 的 unique suffix；不同请求 suffix 不相交，也不与 shared/background keys 冲突，因此 actual reuse 不可能因 key 碰撞高于 planned reuse。
- **C32 是 32 个逻辑 request context**（单一 deque 容器），只有一个 foreground scheduler 线程串行调用 match/insert/release；`--task-pool-size` 只控制后台 load/evict/store task pool，不创建请求线程。
- admission 顺序为 `match → 分配 load targets → 预分配 suffix blocks → commit load`；任一分配/commit 失败整体回滚，无残留 ref/resource。suffix blocks 在 admission 阶段准备好，insert 阶段不再发生 KV allocation。
- load-before-forward：ticket 未完成不能进 forward；一个 ticket pending 不阻塞其他请求 admission。ticket 完成后只有 success 进 READY，failed/cancelled 走 cleanup。
- 每个 READY batch 只 sleep 一次固定 100ms（模拟 forward），不按 batch 大小、长度或 token 缩放；sleep 期间 batch 内请求继续持有 REQUEST refs，scheduler 不发起新 match/insert。`simulated_forward_sleep_ns == forward_batches × 100ms` 是硬性 closure。
- forward 后对 batch 串行执行一次且仅一次 full-path insert，然后释放 matched/load-target/suffix 的 request refs。
- warmup 15s 完整生命周期 → quiesce + pressure check（每个 device pool used ≥ 75%、host used > 0、device heap > 0、warmup completed ≥ 256）→ measured ≥ 60s → deadline 后停止 admission 但 drain 全部已 admission request → finalize 验证 active contexts / pending tickets / task pool / REQUEST ref 全部归零。
- 单 request lifecycle 超过 60s 时 case 失败，进入有界 cancel/drain，仍写出失败结果与资源 snapshot。
- 路径/RNG trace（20k 条）在 setup 前确定性生成，timed region 内不进行 RNG 或 path 生成；workload definition hash 覆盖除 task-pool size 外的固定协议配置，trace hash 覆盖完整请求元数据与 path。正式 tp4/tp8 对照按 `(seed, repetition identity)` 逐组校验唯一变量。

## 输出指标（result.json）

### Tree 生命周期指标

| 指标 | 含义 |
| --- | --- |
| `tree_lifecycle.completed_request_transactions` | 完整 match→forward→insert 请求数（driver 校验 > 0） |
| `tree_lifecycle.completed_base_requests` / `completed_continuation_requests` / `continuation_families_completed` | BASE/CONTINUATION 完成数及完成 continuation 的 family 数；正式 workload 必须覆盖全部 32 个 family |
| `tree_lifecycle.forward_batches` / `forward_requests` | 固定 sleep 的 batch 数 / batch 内请求总数；`forward_requests == completed_request_transactions` 是硬性 closure |
| `tree_lifecycle.simulated_forward_sleep_ns` | `forward_batches × 100ms`；与 `forward_batches` 的 closure 由 driver 校验 |
| `tree_lifecycle.pressure_ready` | warmup 后压力观察值；不作为 repetition 的硬 PASS 条件 |
| `tree_lifecycle.failed_requests` / `final_active_requests` / `final_pending_load_tickets` / `final_pending_tasks` / `final_request_ref_blocks` | 失败请求数；finalize 后四类运行态/资源残留必须全为 0 |
| `tree_lifecycle.drain_timeouts` | setup/warmup/measured/finalize 有界 drain 的超时次数；必须为 0 |
| `tree_lifecycle.unexpected_extra_match_count` | actual reuse 超过 planned reuse 的观察计数；用于解释 reuse，不单独判失败 |
| `resolved_config.logical_concurrency`（=32）/ `foreground_scheduler_threads`（=1）/ `task_pool_size_resolved` | C32 是 context 数而非线程数；task pool 是唯一变量 |
| `resolved_config.workload_definition_hash` / `trace_hash` | 固定 workload 协议与确定性 trace 的 FNV-1a hash，用于 tp4/tp8 逐 repetition 唯一变量校验 |
| `match/insert/load_commit/match_to_ready_latency_ns_min/p50/p99/max/avg` | 各阶段时延（ns）；match/insert 不含固定 forward sleep |
| `planned_reuse_blocks_per_request` / `actual_matched_depth_per_request` | planned（分布采样）与实际复用深度；actual 可以低于 planned |
| `planned_reuse_blocks_*` / `actual_matched_depth_blocks_*` / `actual_minus_planned_reuse_blocks_*` | measured 成功请求的 planned、actual 与差值 min/p50/p99/max/avg 分布 |
| `completed_requests_per_family_*` / `completed_epochs_per_family_*` / `completed_generation_*` | measured 成功请求的 family、epoch 和 generation 分布摘要 |
| `device_matched_blocks_per_request` / `host_matched_blocks_per_request` | 每请求 device/host 匹配 block 数 |
| `insert_path_keys_per_request` / `insert_new_nodes_per_request` | 完整 path 长度 / admission 预分配的唯一 suffix block 数 |
| `loads_committed/succeeded/failed/cancelled`、`load_target_allocation_failed`、`suffix_allocation_failed`、`load_commit_failed` | load 与 admission 各终态计数；正式结果必须全为 0（除 committed/succeeded 外） |
| `active_requests_peak` / `waiting_requests_peak` / `loading_requests_peak` / `load_tickets_pending_peak` | 生命周期峰值；`active_requests_peak ≤ 32`，pending peak 作为异步 load 行为观察值，不作为硬 PASS 门槛 |
| `ready_batch_size_avg/max`、`scheduler_no_ready_wait_ns`、`held_request_blocks_peak`、`forward_batches`、`completed_request_transactions` | batch 形状、无 READY 时的 load 轮询等待、跨 forward 持有的 REQUEST blocks 峰值 |
| `benchmark_request_transactions_per_second` | completed transactions / 实际 `measured_ns`；**是 benchmark 口径，不是线上 wall TPS** |
| `pressure_ready`、`warmup.*`、`final.*`、`pool.<name>.*` | warmup 后压力快照与 finalize 快照（含 REQUEST ref 清零） |
| `phases_ns.setup/warmup/measured/finalize` | 各阶段实际时长；measured ≥ 60s（由 driver 按 resolved config 校验） |

**tree 场景 transaction/s**（同固定配置比较用）：`metrics.completed_request_transactions / phases_ns.measured * 1e9`。报告表由 `generate_report.py` 自动计算；任务池对照只比较 load readiness、scheduler no-ready wait、cache 时延与 transaction/s，不推导线上 TPS。

transfer 场景额外输出 `operations_per_second`、`logical_throughput_bytes_per_second`、`total_bytes_transferred`、每方向明细（见 cases 文档）。

## Repetition、磁盘和报告

正式 profile 默认每个 case 只运行 1 个 repetition，以控制整套测试耗时；driver 的 `--process-repetitions` 默认值也是 `1`。只有明确需要稳定性或统计分布分析时，才手工提高该参数。

`--perf record` 不增加 repetition 数，但会为每个代表 case 额外启动 1 个 profiling process。Tree/Transfer runner 使用同一 marker 协议：`PROFILE_ATTACH_READY`（driver 在此 attach perf）→ 预留 2s attach 窗口（不计入 measured）→ `MEASURE_START`（measured timer 才开始）。perf 使用 DWARF call graph，生成 `perf.data`、`perf.folded`、`flamegraph.svg`、`perf_summary.txt` 和栈质量摘要 `stack_quality.txt`；工具目录可用 `--flamegraph-tools-dir` 指定，也可通过 `FLAMEGRAPH_DIR`、`~/FlameGraph` 或 `~/FlameGraph-master` 自动发现。正式 HTML 报告必须链接这些产物，不能只显示 perf 状态。

需要离线手工复现火焰图时（例如拿到 `perf.data` 后重新出图），步骤与 driver 内部完全一致：

```bash
wget http://search-ad.oss-cn-hangzhou-zmf.aliyuncs.com/xingyu/FlameGraph-master.zip
unzip FlameGraph-master.zip

perf script -i perf.data &> perf.unfold
./FlameGraph-master/stackcollapse-perf.pl perf.unfold &> perf.folded
./FlameGraph-master/flamegraph.pl perf.folded > perf.svg
```

这里生成的 CPU 火焰图只反映 on-CPU 时间；线程等待 mutex/futex、条件变量、IO 或调度的时间不会出现在 CPU 火焰图里。BlockTreeCache 的固定 off-CPU 补充流程见下节；CUDA 异步执行与 stream 依赖仍需使用 nsys/NVTX timeline。

**perf 常见故障排查**：

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| perf.data 为空或很小（<1KB） | `perf_event_open` 被 seccomp 或权限策略拦截 | seccomp 放行，并确认宿主 `kernel.perf_event_paranoid` / perf capability 允许采集 |
| 火焰图全为 `[unknown]` | binary 被 strip、debug info 缺失或 `perf script` 用户不匹配 | 确认 base 的 `-g --strip=never` 生效，并以 perf.data 属主运行 `perf script` |
| `"native process did not announce a profiler attach marker"` | runner 未输出任何 marker | 新增/修改 runner 时确保 measured 前依次打印 `PROFILE_ATTACH_READY` 并 sleep 2s，再打印 `MEASURE_START` |
| 符号不全，大量 `[unknown]` | 以 root/sudo 执行 `perf script` | 以 perf.data 属主身份执行（不要 sudo，否则 HOME=/root 找不到 buildid cache） |
| 火焰图样本太少 | 测量窗口或采样频率不足 | 提高 `--perf-frequency`；Tree measured 固定 60s，不再用 `--min-measured-seconds` 调整 |

### CPU perf 的执行位置与容器 fallback

CPU perf 不要求 host PID namespace，也不依赖 BCC sidecar；collector 只需使用对自身可见的目标 PID。标准 driver 会在同一 namespace 内启动 `perf` 和 native process，因此宿主机无法承担编排时，可以直接进入 benchmark 容器运行前文同一条命令。CPU perf 结果不因 off-CPU 条件不足而跳过。

| 执行位置 | CPU perf | BCC off-CPU |
| --- | --- | --- |
| 宿主机 | 支持；需具备 benchmark、`perf` 和 perf-event 权限 | 推荐作为控制平面；宿主 root BCC 可直接采集 |
| 普通 benchmark 容器（private PID） | **支持，也是宿主无法编排时的标准 fallback** | 不支持标准 `offcputime -p` 流程 |
| host-PID privileged 容器 | 支持 | 仅在下节全部条件通过时支持 |

容器内采集 CPU perf 时，`perf` 必须已安装，seccomp 必须放行，宿主 `kernel.perf_event_paranoid` 与进程权限的组合也要允许 `perf_event_open`（需要 capability 时用 `CAP_PERFMON`，旧内核用 `CAP_SYS_ADMIN`）；binary 使用普通 opt/debug-info 构建，并在该环境中提供 FlameGraph 工具。`perf script` 应由 `perf.data` 属主执行。具体故障判断见上表，不需要为了 CPU perf 重建为 `--pid=host` 容器。

### Off-CPU 固定采集流程（仅 `tree_online_high_variation_c32`）

#### 策略和能力边界

- 本节是正式 profile 的**默认必做步骤**，不是用户额外点名后才执行的可选增强。只有上一节定义的环境预检失败才允许 skip，并必须保留具体失败证据。
- 固定顺序为：有效性 repetition → 独立 CPU perf process → 独立 off-CPU process。不要在同一个 benchmark process 上同时挂 CPU perf 和 BCC。
- off-CPU 只采 `tree_online_high_variation_c32`（task pool 4、固定 seed、固定 workload），在 `MEASURE_START` 后立即 attach，固定采集 15 秒；runner 在 `MEASURE_START` 前已预留 2s `PROFILE_ATTACH_READY` attach 窗口，off-CPU 从 `MEASURE_START` 开始 attach，因此不会把 profiler attach sleep 计入 folded stacks。
- 首选“宿主机编排 + benchmark 容器 + BCC profiler sidecar”；宿主已有可用的 root BCC 时，也可不启 sidecar。
- 无法从宿主编排时，合格的 host-PID privileged 容器也可以承担编排/采集；条件与 sidecar 完全相同。任一条件无法验证，只跳过 off-CPU，保留 benchmark 和 CPU perf。
- 不允许用容器 PID、`--pid=container:<target>`、全系统无过滤采集或 unknown-heavy SVG 作为降级结果。

`offcputime` 在内核中按 stack id 聚合阻塞时长，输出前再利用 `/proc/<host-tgid>/maps`、ELF 符号表和 debug info 将地址解析为函数名，这一步即 BCC 符号化。因此 `-p` 必须传宿主 TGID，目标 ELF/动态库必须可达，而且目标进程要存活到 `offcputime` 完成输出。sidecar 只负责加载 BPF、读取聚合结果和符号化，不参与 benchmark 逻辑，也不需要 GPU。

collector 必须同时满足：

1. 位于 host PID/cgroup namespace，能唯一获得宿主 TGID，并读取 `/proc/<TGID>/status` 和 `maps`。
2. 以宿主 root 运行，或容器以 `--privileged --security-opt seccomp=unconfined` 创建。
3. 可访问宿主 tracefs/debugfs/BTF/bpffs，以及与 `uname -r` 精确对应的 `/lib/modules` 和 `/usr/src/kernels`。
4. 已安装 BCC，`offcputime -K 1` smoke 能成功加载 BPF。
5. binary 未 strip、包含 debug info，且 `/home` 或目标 root 中的 ELF/动态库可读。普通 binary 不再全局保留 frame pointer，
   因而 BCC off-CPU 用户栈仅为 best-effort；质量门禁不通过时跳过该产物，不影响 benchmark 与 DWARF CPU perf。

普通 private-PID 容器通常只满足 CPU perf 条件，不满足第 1 项；容器创建后也不能在线修改 PID mode。若合格的 collector 容器还要自主启动另一个 benchmark 容器，还需 Docker CLI 和 `/var/run/docker.sock`；若目标与 collector 同容器，还需完整 CUDA/GPU 和 benchmark 运行环境。

#### profiler sidecar：创建与预检

sidecar 可以复用；已有同名容器时先验收，不要盲目删除。`BENCH_USER` 指共享 `/home` 中的代码属主；collector/orchestrator 以 root 运行时也必须显式填写该用户。下面以 benchmark image 为基底：

```bash
BENCH_USER="<benchmark user>"
BENCH_CONTAINER="<cuda benchmark container>"
PROFILER_CONTAINER="${BENCH_USER}_offcpu_profiler"
PROFILER_IMAGE="$(docker inspect --format '{{.Config.Image}}' "$BENCH_CONTAINER")"

docker run -d \
  --name "$PROFILER_CONTAINER" \
  --pid=host --cgroupns=host --net=host \
  --privileged \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  -v /home:/home \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  -v /sys/kernel/btf:/sys/kernel/btf:ro \
  -v /sys/fs/bpf:/sys/fs/bpf \
  "$PROFILER_IMAGE" sleep infinity

docker exec --user root "$PROFILER_CONTAINER" \
  yum install -y --setopt=install_weak_deps=False \
    bcc bcc-tools python3-bcc
```

每次采集前执行下面的最小预检；任何失败都记为 `off-CPU: skipped (<reason>)`：

```bash
test "$(docker inspect --format '{{.State.Status}}' "$PROFILER_CONTAINER")" = running
test "$(docker inspect --format '{{.HostConfig.PidMode}}' "$PROFILER_CONTAINER")" = host
test "$(docker inspect --format '{{.HostConfig.CgroupnsMode}}' "$PROFILER_CONTAINER")" = host
test "$(docker inspect --format '{{.HostConfig.Privileged}}' "$PROFILER_CONTAINER")" = true
docker inspect --format '{{json .HostConfig.SecurityOpt}}' "$PROFILER_CONTAINER" \
  | rg -q 'seccomp=unconfined'

docker exec --user root "$PROFILER_CONTAINER" \
  awk '/^NSpid:/ {found=1; if (NF != 2) exit 1} END {if (!found) exit 1}' \
    /proc/self/status

docker exec --user root "$PROFILER_CONTAINER" bash -lc '
  set -e
  test -x /usr/share/bcc/tools/offcputime
  test -r /sys/kernel/btf/vmlinux
  test -d "/lib/modules/$(uname -r)"
  test -d "/usr/src/kernels/$(uname -r)"
  test "$(findmnt -nro FSTYPE --target /sys/kernel/tracing)" = tracefs
  test "$(findmnt -nro FSTYPE --target /sys/kernel/debug)" = debugfs
  test "$(findmnt -nro FSTYPE --target /sys/fs/bpf)" = bpf
  /usr/share/bcc/tools/offcputime -K 1 >/dev/null
'
```

#### 采集 `tree_online_high_variation_c32`

CPU perf 先用前文 driver 命令独立完成；若在容器内执行，可将 `--output-dir` 设为 `$PROFILE_ROOT/cpu`。off-CPU 使用另一个 native process（只传 common options 与 `--task-pool-size`）：

```bash
BENCH_USER="<benchmark user>"
REPO="/home/$BENCH_USER/RTP-LLM/github-opensource"
FLAMEGRAPH_DIR="/home/$BENCH_USER/FlameGraph-master"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="/home/$BENCH_USER/profile_results/block_tree_cache/tree_online_high_variation_c32_$RUN_ID"
OFFCPU_DIR="$PROFILE_ROOT/offcpu"
BINARY="$REPO/bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_gpu_benchmark"
MODEL_PROFILE="$REPO/rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/deepseek_v4_pro_fp8_tp1_cp1.json"

mkdir -p "$OFFCPU_DIR"

docker exec \
  --user "$BENCH_USER" \
  --workdir "$REPO" \
  --env LD_LIBRARY_PATH=/opt/conda310/lib \
  "$BENCH_CONTAINER" \
  "$BINARY" tree \
    --cuda-device 0 --max-device-memory-fraction 0.8 \
    --model-profile "$MODEL_PROFILE" --seed 42 --repetition-id 0 \
    --output-json "$OFFCPU_DIR/result.json" \
    --task-pool-size 4 \
  >"$OFFCPU_DIR/stdout.txt" 2>"$OFFCPU_DIR/stderr.txt" &

BENCH_EXEC_PID=$!
ATTACH_DEADLINE=$((SECONDS + 900))
until rg -q '^MEASURE_START$' "$OFFCPU_DIR/stdout.txt"; do
  kill -0 "$BENCH_EXEC_PID" 2>/dev/null || {
    wait "$BENCH_EXEC_PID" || true
    exit 2
  }
  ((SECONDS < ATTACH_DEADLINE)) || {
    wait "$BENCH_EXEC_PID" || true
    exit 2
  }
  sleep 0.1
done
```

marker 出现后，从宿主进程表按本次唯一的 `result.json` 路径解析 TGID；不要使用业务容器内 PID：

```bash
mapfile -t HOST_TGIDS < <(
  docker top "$BENCH_CONTAINER" -eo pid,args |
    awk -v marker="$OFFCPU_DIR/result.json" 'index($0, marker) {print $1}'
)
((${#HOST_TGIDS[@]} == 1)) || {
  wait "$BENCH_EXEC_PID" || true
  exit 2
}

HOST_TGID="${HOST_TGIDS[0]}"
test -r "/proc/$HOST_TGID/status"
test -r "/proc/$HOST_TGID/maps"
printf '%s\n' "$HOST_TGID" >"$OFFCPU_DIR/host_tgid.txt"

docker exec --user root "$PROFILER_CONTAINER" \
  /usr/share/bcc/tools/offcputime \
    -d -f -p "$HOST_TGID" --stack-storage-size 16384 15 \
  >"$OFFCPU_DIR/offcpu.folded" \
  2>"$OFFCPU_DIR/offcputime.stderr"

# BCC 已完成输出和符号化；此刻目标必须仍然存活。
test -r "/proc/$HOST_TGID/maps"
wait "$BENCH_EXEC_PID"
```

宿主 root BCC 可将最后一条 `docker exec ... offcputime` 换成：

```bash
sudo /usr/share/bcc/tools/offcputime \
  -d -f -p "$HOST_TGID" --stack-storage-size 16384 15 \
  >"$OFFCPU_DIR/offcpu.folded" \
  2>"$OFFCPU_DIR/offcputime.stderr"
```

`-f` 已直接输出 FlameGraph folded 格式，最后一列为微秒；不要再运行 `stackcollapse-perf.pl`。采集命令失败、目标提前退出或 benchmark 失败时，保留 stderr/result 并跳过出图。

#### 质量验收、出图与报告

只有 benchmark 生命周期健康且采样质量同时满足以下条件才生成并上传 SVG：

- `result.json` 为 `completed`，measured ≥ 60 秒，BASE/CONTINUATION 均完成且全部 32 个 family 完成 continuation，hard-failure 计数与 drain timeout 为 0，final active requests / pending tickets / task-pool pending tasks / REQUEST ref delta 均为 0。`pressure_ready`、unexpected extra match 和 pending peak 只作为观察指标。
- `offcpu.folded` 非空，并包含 `rtp_llm::` 或 `BlockTree` 业务符号。
- 目标存活到 BCC 符号化完成；`[Missed User Stack]` 的加权时间占比 ≤ 1%。

```bash
test -s "$OFFCPU_DIR/result.json"
test -s "$OFFCPU_DIR/offcpu.folded"
rg -q 'rtp_llm::|BlockTree' "$OFFCPU_DIR/offcpu.folded"

awk '
  { total += $NF }
  /\[Missed User Stack\]/ { missed += $NF }
  END {
    if (total <= 0) exit 2
    ratio = missed / total
    printf("total_us=%.0f missed_us=%.0f missed_ratio=%.6f\n",
           total, missed, ratio)
    if (ratio > 0.01) exit 3
  }
' "$OFFCPU_DIR/offcpu.folded" >"$OFFCPU_DIR/offcpu_quality.txt"

perl "$FLAMEGRAPH_DIR/flamegraph.pl" \
  --colors=io \
  --title='BlockTreeCache tree_online_high_variation_c32 Off-CPU Flame Graph' \
  --countname=us \
  <"$OFFCPU_DIR/offcpu.folded" \
  >"$OFFCPU_DIR/offcpu_flamegraph.svg"

test -s "$OFFCPU_DIR/offcpu_flamegraph.svg"
```

固定 100ms forward sleep 本身就是有意制造的 off-CPU 时间，task pool 空闲时的条件变量等待也是预期背景；`offcputime -p` 会同时采到这些栈。因此：

- **保留未经删除或过滤的原始 `offcpu.folded`**，不允许用正则过滤后的图代替原始产物；
- 同时核对 `simulated_forward_sleep_ns`、`scheduler_no_ready_wait_ns` 与 load readiness 指标，把 `sleep_for`/`nanosleep` 对应的 forward 模拟等待标为 **expected**，不得归因成 BlockTreeCache 退化；
- 只有额外出现的 cache 锁等待、load/evict completion 争用或 scheduler no-ready wait 才作为优化线索。

闭源 CUDA driver 或系统库中少量 `[unknown]` 可以接受，前提是业务栈和 missed 比例达标。宿主 Perl 缺 module 时，可在共享同一 `/home` 的 benchmark 容器内执行相同的 `flamegraph.pl` 命令。

正式报告在“火焰图与采样质量”中链接：

```text
profile/tree_online_high_variation_c32/offcpu_<RUN_ID>/offcpu_flamegraph.svg
profile/tree_online_high_variation_c32/offcpu_<RUN_ID>/offcpu.folded
profile/tree_online_high_variation_c32/offcpu_<RUN_ID>/offcpu_manifest.json
```

同时写 `offcpu_manifest.json`，记录 case、commit、binary SHA、resolved config hash、seed、host TGID、BCC duration、目标存活检查和符号质量；它是补充 profile artifact，不得伪装成 suite 的有效 repetition。若 off-CPU process 与原 suite 的 commit 或 binary SHA 不同，标为“独立补充采集”，不要修改原 `suite_manifest.json`；条件不满足时明确写 `off-CPU: skipped (<reason>)`，不要上传 SVG。

driver 为每个 repetition 创建独立 result 目录、disk 目录和 vmstat 窗口。buffered case 在 measured process 退出后单独执行 filesystem drain、记录 drain 时间，再采样 after 并清理该 repetition 文件。

`result.json` 在启动 native process 前删除；只有本次生成、`status=completed` 且 component 对应的 closure invariants 通过的结果才标记 `valid=true`：Tree 校验 lifecycle/transaction closure、BASE/CONTINUATION family 覆盖、hard-failure/drain 与 final-zero；Transfer 校验 operation/working-set/strategy closure。

HTML 报告只读取 `profile/suite_manifest.json` 中的 valid repetitions，输出 median、MAD、min/max 和样本数；数值一律 human-readable：时延按 `ns`/`us`/`ms`/`s` 自适应单位、整数加千分位、禁用科学计数法，n=1 时只输出单值（格式约定见 [docs/report_template.md](docs/report_template.md) 文首）：

```bash
python3 rtp_llm/cpp/cache/block_tree_cache/benchmark/generate_report.py \
  --output-dir /tmp/btc_profile \
  --output /tmp/btc_profile/index.html
```

Tree 报告先解释 block/token/payload、20 档请求长度及 BASE（新会话）/CONTINUATION（续写）构造，再用用户口径展示完整请求生命周期数量与 req/s、请求组成、命中深度、关键时延和结束清理。`pressure_ready`、dependency skip 等内部诊断字段只在“主要观察”中翻译说明，不作为抽象表头，也不把非失败的水位观察显示成红色失败状态。tp4/tp8 专项结果按 `task_pool_size_resolved` 显式分组，并逐 repetition 校验除 pool size 外的 resolved config、profile、seed/repetition、trace、binary SHA 与代码 commit，不一致则拒绝比较。火焰图表只列实际生成 profiling artifact 的 case；off-CPU 产物以独立 artifact 行展示，不计入 repetition 聚合。

最终报告以 `index.html` 呈现；[报告模板](docs/report_template.md) 是仓库内的格式规范，不是另一份 Markdown 交付物。HTML 必须遵循该模板的章节顺序与信息边界，结论和 suite 完整性状态放在最前面，不得为了展示效果自行删减模板要求的关键内容。

GPU、PCIe、磁盘、binary/profile SHA 和代码 commit 来自 suite manifest 的实际采集。磁盘配置按 benchmark 进程 mount namespace 采集；raw manifest 保留容器内可见的 target/source/fstype 与容量，HTML 环境表把它们压成一行，不输出完整 overlay lowerdir/upperdir，也不推测宿主机物理块设备。要测宿主指定磁盘，先将该目录 bind mount 到 benchmark 容器，再把 `--disk-root` 指向容器内路径。未配置同机硬件基线阈值时，自动报告只展示事实并标记“待分析”，不会输出“接近硬件上限”等因果结论。没有 off-CPU 产物时，报告会显式展示 manifest 或 `report_metadata.offcpu_status` 中的跳过原因，不再静默省略该小节。

`pgpgin`/`pgpgout` 是系统累计量在单 repetition 窗口内的差值，只能作为 ancillary signal，不能精确归因到单进程、单方向或单次 IO。

## 结果上传 OSS（可选）

跑完 suite 并生成 `index.html` 后，把 `index.html` + `profile/` 一起上传，路径带时间戳前缀，多人多次互不冲突：

```bash
PREFIX="$(whoami)/$(date +%Y%m%d_%H%M%S)"
# 或多人共享目录：oss://search-ad/rtp-llm/block_tree_cache_benchmark/<date_time>_<tag>/
ossutil cp -r /tmp/btc_profile/ oss://search-ad/$PREFIX/

# 校验（HTTP 访问）
curl -sI "http://search-ad.oss-cn-hangzhou-zmf.aliyuncs.com/$PREFIX/index.html" | head -3
```

**常见坑**：
- `ossutil cp` 覆盖**已存在**的同路径对象时会交互式询问 `overwrite ...? (y or N)`；非 TTY 下 EOF 被当作 N，**静默跳过上传**。覆盖上传必须加 `-f`，并上传后用 `curl -sI` 核对。
- 删除旧产物用 `ossutil rm -r -f <prefix>/`（不加 `-f` 会交互确认）。

## 常见坑速查表

| 坑 | 症状 | 解决 |
| --- | --- | --- |
| 未设置 `LD_LIBRARY_PATH` | `libpython3.10.so.1.0: cannot open shared object file` | `export LD_LIBRARY_PATH=/opt/conda310/lib` |
| DWARF 火焰图出现大量 `[unknown]` | debug info、build-id cache 或 perf.data 属主不匹配 | 确认 base 的 `-g --strip=never`，并以采集用户运行 `perf script` |
| 临时切换到全新的 Bazel `output_user_root` | 重新拉取外部仓库并分析约 2.3 万个传递 action，即使源码只改一处也会等待很久 | 每个 worktree 固定复用自己的 cache root；正式测试前先完成一次目标预构建，不在计时窗口中冷启动 cache |
| Bazel 报 pip repository `used before defined` cycle | internal `rtp_deps`/`arch_config` 与当前开源 WORKSPACE 定义不同步 | 先同步两套依赖配置；临时验证可显式 `--override_repository=rtp_deps=<github-opensource/deps>` 与 `--override_repository=arch_config=<github-opensource/arch_config>`，并把 override 写入报告复现命令 |
| 直接跑 binary 时照抄 driver 的 case 参数 | 参数不识别或静默失败 | binary 参数见 `--help`；Tree 只接受 common options 与 `--task-pool-size`，旧 synthetic options 会被拒绝 |
| 把 C32 当成 32 个线程 | 误读结果 | C32 是 32 个逻辑 request context，foreground scheduler 恒为 1 线程 |
| 忘记 `--disk-root` | disk case 记为 `skipped_no_disk` | 添加 `--disk-root <真实磁盘目录>` |
| 容器内 `/tmp` 显示为 `overlay` | 报告只能看到容器 mount namespace，无法由 overlay 反推宿主物理盘 | 若需关联宿主真实磁盘，把宿主磁盘目录 bind mount 进容器，并对该容器路径运行；报告保留 container-visible source/fstype/容量 |
| 旧 perf/core/disk working set 占满 `/tmp` | buffered case 空间不足、结果受磁盘余量影响或中途失败 | 开跑前只清理明确属于旧 benchmark 的目录/core/perf 文件；每个 repetition 完成 drain 后由 driver 删除自己的 disk 目录 |
| 修改 `benchmark_cases.py` 后未重建 driver | 新 case 参数不生效 | 重新构建 `block_tree_cache_benchmark_driver` |
| 修改 C++ runner 后未重建 binary | 行为与源码不一致 | 重新构建 `block_tree_cache_gpu_benchmark` |
| device↔disk case 报 `staging pool exhausted` | 显式 staging 容量不足以承载当前提交形态 | benchmark 不会静默放大配置；调整 `--device-disk-staging-block-count`、batch 或并发度，并核对结果中的 staging count/capacity |
| case 失败但 result.json 无任何指标 | warmup/pilot 阶段失败被静默丢弃 | 已修复为写入 `warmup.*` / `pilot.*` first_error；旧 binary 需重新构建 |
| tree 结果 `pressure_ready=false` | warmup 后未达到预期压力形态 | 查看 `warmup.*` / `pool.*` 快照解释容量或时长；它是观察项，不会单独使 repetition 无效 |
| 某张 perf 火焰图样本很少 | 短 case 或大量 off-CPU 等待导致 on-CPU samples 不足 | 在报告标明样本数并避免强热点归因；必要时延长独立 profiling process，不伪装成 suite repetition |
| suite 全部 case 秒挂、stderr 报 libpython | 见第一行 | 设置 `LD_LIBRARY_PATH` 后重跑 |
