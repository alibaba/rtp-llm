# BlockTreeCache Benchmark

本目录提供 BlockTreeCache 的 Tree metadata microbenchmark 与 Device/Host/Disk transfer benchmark。当前实现以 workload 可核对、失败可传播、repetition 相互独立为前提；整改前的 buffered 大工作集与 round-trip 数值不能作为基线。

> [!IMPORTANT]
> 运行本 benchmark 前必须先阅读并遵循本 README。不建议参考或套用 benchmark 目录外的通用 skill、测试流程或构建参数；除非本文明确引用，构建配置、运行方式、case 定义和结果判定均以本目录文档为准。外部说明与本文冲突时，以本文为准。

## 设计边界

- C++ binary 只有 `tree`、`transfer` 两个子命令；suite/case registry 由 `benchmark_cases.py` 统一维护。
- Tree case 是 scaled/flattened metadata microbenchmark。它保留 GroupSet 类型与 payload 压力，但不声称复刻 profile 的 member fan-out、device-only group 或真实 SWA topology。
- Transfer 的 device 侧按 profile member/layer 构造真实 copy tile；worker slot 只按并发分配，host/disk working set 独立寻址，不按 working set 大小扩张 GPU staging 内存。device↔disk staging lease 数至少取并发数，避免 worker 超订导致 `staging pool exhausted`。
- mixed case 在同一 measured window 内交替执行两个方向。展示的“混合总吞吐”是该窗口内两个方向成功字节数之和除以墙钟时间，不是两个单方向峰值的相加。
- tier round-trip 已删除。demote/load 生命周期正确性继续由现有功能测试看护。
- 计时边界：阶段为 `bootstrap → profile load → allocation → setup → warmup → pilot → measured → sync/drain → teardown`，只有 `measured`（及界定完成时刻的 sync/drain）计入指标；perf 运行是独立进程，其 wall time 不进入统计。
- 内存模型：常驻显存由“同时存活的 blocks × 单 block payload”决定，而非 Tree 节点总数。profile 提供 `model_sized`（真实 payload）与 `scaled`（payload 按 1/256 缩放）两种档位，资源预检不足时直接终止，不自动缩小命名 case。

## Case 矩阵

- smoke：2 个，只运行最小 Tree 与 D2H Transfer 端到端路径。
- profile：14 个，包括 2 个 Tree、4 个 Device↔Host copy-strategy 对照、4 个 Device↔Disk、4 个 Host↔Disk。
- buffered profile working set：`full_context=32768` blocks，`swa=4096` blocks。每次 measured window 至少完整访问一轮 addressable working set。

权威清单见 [docs/benchmark_cases.md](docs/benchmark_cases.md)。

## 构建

```bash
cd github-opensource

# 普通构建（跑指标）
bazelisk build -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_gpu_benchmark \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_driver

# perf 构建（火焰图需要：不 strip + frame pointer；只重构 gpu_benchmark 即可）
bazelisk build -c opt --config=cuda13 --config=sm8x --config=block_tree_benchmark_perf \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_gpu_benchmark
```

产物在 `bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/`。

**常见坑**：
- perf 构建必须用 `--config=block_tree_benchmark_perf`（等价于 `--copt="-fno-omit-frame-pointer" --strip="never"`），否则火焰图函数名显示为裸地址。
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
# 日常 UT 看护（需 GPU）：一条 smoke 命令即可
bazelisk test -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_smoke_test

# 改动 benchmark 代码时的针对性回归（纯逻辑、无需 GPU、秒级）：
# workload 调度不变量、tree 请求模板语义、case registry、报告生成与格式
bazelisk test -c opt --config=cuda13 --config=sm8x \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:transfer_benchmark_workload_test \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:tree_workload_generator_test \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_cases_test \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_report_test \
  //rtp_llm/cpp/cache/block_tree_cache/transfer/test:per_rank_transfer_engine_device_host_test
```

日常 UT 看护只需跑上面的 smoke test：它已经覆盖 binary/driver 启动、case registry 与 profile 加载、最小 Tree 与 D2H Transfer 的端到端路径。四个纯逻辑测试无需 GPU、秒级完成，只在改动 benchmark 自身代码时作为针对性回归运行；其中 `block_tree_cache_benchmark_report_test` 专门看护报告生成与数值格式，因为格式回归（例如科学计数法）不会让 smoke test 失败。

### 用 driver 跑 suite（推荐）

```bash
export LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH

# 正式 profile；默认每 case 只运行 1 次，约 25-35 分钟
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --process-repetitions 1 \
  --output-dir /tmp/btc_profile \
  --disk-root /path/to/benchmark_disk

# profile + perf 收集（需 perf 构建，见「构建」；自动发现 ~/FlameGraph-master）
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --process-repetitions 1 \
  --output-dir /tmp/btc_profile \
  --disk-root /path/to/benchmark_disk \
  --perf record

# 只复测两个 Tree case；每项 1 次有效性 repetition，并各启动 1 次 profiling process
./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_benchmark_driver \
  --suite profile \
  --case tree_stress_100k,tree_stress_100k_single \
  --process-repetitions 1 \
  --perf record \
  --flamegraph-tools-dir /path/to/FlameGraph \
  --output-dir /tmp/btc_tree_profile
```

默认是严格 suite：任一 failed、partial、skipped 或 required perf 失败都会返回非零。仅调试时可显式传 `--allow-incomplete`。缺少 `--disk-root` 时 disk case 仍保留在 manifest，并记为 `skipped_no_disk`。

### 直接跑 binary（调试用）

```bash
LD_LIBRARY_PATH=/opt/conda310/lib ./bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_gpu_benchmark tree \
  --model-profile rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/deepseek_v4_pro_fp8_tp1_cp1.json \
  --payload-mode scaled \
  --tree-node-count 100000 --max-path-length 1000 --tree-branching-factor 16 \
  --initial-min-path-length 128 --initial-max-path-length 768 \
  --continuation-ratio 0.7 --fork-ratio 0.2 \
  --fork-reuse-min-ratio 0.25 --fork-reuse-max-ratio 0.9 \
  --hot-path-ratio 0.2 --active-path-limit 4096 \
  --append-length 32 --inserts-per-match 4 --operation-trace-count 20000 \
  --steady-threads 8 --warmup-seconds 10 --min-measured-seconds 30 \
  --output-json /tmp/tree_result.json
```

> 注意：binary 的参数名与 driver 的 case 参数名**不同**。driver 会做翻译（例如 driver 的 `--disk-root` 对每个 repetition 展开成 binary 的 `--disk-path <rep 子目录>`）；手动跑 binary 时不要照抄 case 参数，直接看 `--help`。

## Transfer 语义与结果核对

`--transfer-operation-count` 是 measured window 的全局总操作数，不是每 worker 操作数。pilot 只允许向上扩大：

```text
final_operations = max(requested, calibrated, working_set_blocks * direction_count)
```

每个 logical coordinate 的方向由同一 worker 顺序执行，并共享 working-set index：

```text
worker_slot_index = worker_id
logical_coordinate = operation_index / direction_count
working_set_index = permutation(logical_coordinate) % working_set_blocks
```

正式结果必须满足：

```text
requested = attempted = succeeded + failed
failed = 0
requested working set = addressable working set = visited working set
```

result 还记录每个 tier 的 capacity/allocated/addressable blocks、每方向 attempted/succeeded/failed、首个错误、requested/actual copy strategy 和是否 wrap-around。显式 `batch`/`staged-sm` 会在 benchmark 配置中关闭另一个优化路径，并由 benchmark-only recorder 记录 measured window 实际命中的策略；任何 fallback 或 mixed 命中都会使 case 无效。

所有 host/device block 在运行前清零。mixed disk case 的 read 跟随同 coordinate 的 write；如果方向顺序无法提供该前置条件，setup 会预填完整 disk working set。warmup、pilot、measured 使用连续 coordinate cursor，不从 offset 0 重放。

## Tree 语义

- 初始树由 128–768 keys 的变长路径构成；路径通过 continuation/fork/cold 混合产生共享前缀，完整路径上限为 1000。
- 每个稳态事务先 match，再执行 4 次 insert；每步在完整 path 后追加 32 keys，且只为 cache 中尚不存在的 suffix 分配资源。fork/cold 的第一次 insert 还会补齐 match 未命中的请求 suffix，所以实际新增 nodes 可以大于 32；后续 insert 通常新增 32。因此“insert path 长度”和“本次新增 node 数”是两个不同指标。
- continuation 完整 match 候选路径；fork 复用候选路径的随机前缀后分叉；cold 使用全新 key space。普通候选按 epoch 无放回抽样，hot subset 可按配置概率重复命中，active pool 有界。
- 每次成功 insert 后的最终路径进入候选池。路径/RNG trace 在 warmup 和 measured 计时区外预生成；trace 不循环重放，耗尽会使 case 失败。
- 结果按 continuation/fork/cold 输出实际请求数、平均 match keys、平均匹配深度、平均 insert 完整路径长度、平均新增 nodes，以及 device/host/disk hit 与 miss。
- Tree 的 async task pool 固定为 32，单/多 worker case 不再随请求线程数改变后台执行容量。
- 节点水位由独立低频 sampler thread 采集；worker 热路径不争用 benchmark-side 全局采样锁。
- async load 分别记录 committed、succeeded、failed、cancelled 与 measured 结束时 pending；pending load 在 measured 结束后单独 drain。
- load 数等于 match 的 host 命中数（每次 host 命中触发一次异步 load，把 block 提升到 device）。多/单线程 load 数差异反映稳态下 eviction 把多少 block 推到 host，不是故障。

## 输出指标（result.json）

| 指标 | 含义 |
| --- | --- |
| `insert_latency_ns_min/p50/p99/max/avg` | insert 每次调用时延（ns） |
| `match_latency_ns_min/p50/p99/max/avg` | match 每次调用时延（ns） |
| `load_latency_ns_min/p50/p99/max/avg` | load 提交时延（ns，仅含分配+commit，不含异步传输） |
| `insert_path_keys_per_call` / `insert_new_nodes_per_call` | insert 收到的平均完整路径长度 / 本次实际新增的平均后缀节点数 |
| `match_keys_per_call` | 每个 match 请求的平均 key 数 |
| `match_device_matched_blocks_per_request` / `match_host_matched_blocks_per_request` | 每个 match 请求平均在 device / host 匹配的逻辑 block 数 |
| `scenario.<continuation\|fork\|cold>.*` | 三类状态化事务的请求、匹配深度、insert 形状及分层命中明细 |
| `trace_exhaustions` | 预生成 trace 是否在测量时限前耗尽；非零时 case 失败 |
| `loads_committed/succeeded/failed/cancelled` | 异步 load 各终态计数 |
| `steady_state_node_count_avg/min/max` | 测量窗内树节点水位 |
| `insert_calls` / `match_calls` | 时延样本数（算 ops/s 用） |

**tree 场景 ops/s**（多线程吞吐对比用）：`metrics.insert_calls / phases_ns.measured * 1e9`（match 同理）。报告表由 `generate_report.py` 自动计算。

transfer 场景额外输出 `operations_per_second`、`logical_throughput_bytes_per_second`、`total_bytes_transferred`、每方向明细（见 cases 文档）。

## Repetition、磁盘和报告

正式 profile 默认每个 case 只运行 1 个 repetition，以控制整套测试耗时；driver 的 `--process-repetitions` 默认值也是 `1`。只有明确需要稳定性或统计分布分析时，才手工提高该参数。

`--perf record` 不增加 repetition 数，但会为每个代表 case 额外启动 1 个 profiling process，只采集 `MEASURE_START` 之后的 measured window。它通过 Brendan Gregg FlameGraph 工具生成 `perf.data`、`perf.folded`、`flamegraph.svg` 和 `perf_summary.txt`；工具目录可用 `--flamegraph-tools-dir` 指定，也可通过 `FLAMEGRAPH_DIR`、`~/FlameGraph` 或 `~/FlameGraph-master` 自动发现。正式 HTML 报告必须链接这些产物，不能只显示 perf 状态。

需要离线手工复现火焰图时（例如拿到 `perf.data` 后重新出图），步骤与 driver 内部完全一致：

```bash
wget http://search-ad.oss-cn-hangzhou-zmf.aliyuncs.com/xingyu/FlameGraph-master.zip
unzip FlameGraph-master.zip

perf script -i perf.data &> perf.unfold
./FlameGraph-master/stackcollapse-perf.pl perf.unfold &> perf.folded
./FlameGraph-master/flamegraph.pl perf.folded > perf.svg
```

火焰图只反映 on-CPU 时间；磁盘等待、CUDA 异步传输等 off-CPU 段不会出现在 CPU 火焰图里，需要时补充 nsys/NVTX timeline。

**perf 常见故障排查**：

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| perf.data 为空或很小（<1KB） | `perf_event_open` 被 seccomp 拦截 | 容器需以 `--security-opt seccomp=unconfined` 启动 |
| 火焰图全为 `[unknown]` | binary 被 strip 或缺少 frame pointer | 用 `--config=block_tree_benchmark_perf` 重新构建 |
| `"native process did not announce MEASURE_START"` | runner 未输出 MEASURE_START | 新增/修改 runner 时确保测量阶段前打印 `MEASURE_START` 并 sleep 2s 给 perf attach 留时间 |
| 符号不全，大量 `[unknown]` | 以 root/sudo 执行 `perf script` | 以 perf.data 属主身份执行（不要 sudo，否则 HOME=/root 找不到 buildid cache） |
| 火焰图样本太少 | 测量窗口太短 | 提高 repetition 或降低 `--perf-frequency` |

driver 为每个 repetition 创建独立 result 目录、disk 目录和 vmstat 窗口。buffered case 在 measured process 退出后单独执行 filesystem drain、记录 drain 时间，再采样 after 并清理该 repetition 文件。

`result.json` 在启动 native process 前删除；只有本次生成、`status=completed` 且计数/working-set/strategy invariants 通过的结果才标记 `valid=true`。

HTML 报告只读取 `profile/suite_manifest.json` 中的 valid repetitions，输出 median、MAD、min/max 和样本数；数值一律 human-readable：时延按 `ns`/`us`/`ms`/`s` 自适应单位、整数加千分位、禁用科学计数法，n=1 时只输出单值（格式约定见 [docs/report_template.md](docs/report_template.md) 文首）：

```bash
python3 rtp_llm/cpp/cache/block_tree_cache/benchmark/generate_report.py \
  --output-dir /tmp/btc_profile \
  --output /tmp/btc_profile/index.html
```

最终报告以 `index.html` 呈现；[报告模板](docs/report_template.md) 是仓库内的格式规范，不是另一份 Markdown 交付物。HTML 必须遵循该模板的章节顺序与信息边界，结论和 suite 完整性状态放在最前面，不得为了展示效果自行删减模板要求的关键内容。

GPU、PCIe、磁盘、binary/profile SHA 和代码 commit 来自 suite manifest 的实际采集。未配置同机硬件基线阈值时，自动报告只展示事实并标记“待分析”，不会输出“接近硬件上限”等因果结论。

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
| 未用 perf config 构建 | 火焰图全为 `[unknown]` | 用 `--config=block_tree_benchmark_perf` 构建 |
| 直接跑 binary 时照抄 driver 的 case 参数 | 参数不识别或静默失败 | binary 参数见 `--help`；`--disk-root` 在 binary 层是 `--disk-path` |
| 忘记 `--disk-root` | disk case 记为 `skipped_no_disk` | 添加 `--disk-root <真实磁盘目录>` |
| 修改 `benchmark_cases.py` 后未重建 driver | 新 case 参数不生效 | 重新构建 `block_tree_cache_benchmark_driver` |
| 修改 C++ runner 后未重建 binary | 行为与源码不一致 | 重新构建 `block_tree_cache_gpu_benchmark` |
| device↔disk case 报 `staging pool exhausted` | staging lease 数小于 worker 并发 | 已修复为自动取 `max(配置, 并发)`；旧 binary 需重新构建 |
| case 失败但 result.json 无任何指标 | warmup/pilot 阶段失败被静默丢弃 | 已修复为写入 `warmup.*` / `pilot.*` first_error；旧 binary 需重新构建 |
| suite 全部 case 秒挂、stderr 报 libpython | 见第一行 | 设置 `LD_LIBRARY_PATH` 后重跑 |
