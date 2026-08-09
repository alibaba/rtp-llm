# BlockTreeCache Benchmark 报告模板

> 用途：按此模板输出每次性能测试报告，保证不同人、不同机器、不同时间的结果可对比、可复核。
> 本文是仓库内可评审、可版本管理的 HTML 报告规范，不是另一份 Markdown 交付物；最终报告只以 `index.html` 呈现。
>
> **正式报告标题**：`BlockTreeCache GPU Benchmark 性能报告（<GPU 型号> · <日期>）`，不要使用“报告模板”字样。
> 实际交付时删除本段说明，所有结果必须来自本次 `suite_manifest.json` 中的 valid repetitions；不得复制旧报告中的打样数值或已作废结果。
>
> **自动生成与留白说明**：
>
> - **结构性内容**：环境、Tree/Transfer 指标、带宽对比、perf 产物和 vmstat 表格由 `generate_report.py` 从 manifest 和 `result.json` 生成。
> - **描述性内容**：执行结论、Tree 核心观察、Transfer 带宽与瓶颈分析、采样质量解读、vmstat 解读、限制与后续建议、产物清单由报告作者结合本次数据填写。这些内容体现分析价值，不得照抄示例结论。
> - 生成的 HTML 是报告初稿。正式发布前必须补齐所有“留白”，并保持本模板的章节顺序与信息边界；结论和 suite 完整性状态优先展示，不得自行删减关键项。

### 数值格式约定

- 时延按量级自适应单位：`ns` / `us` / `ms` / `s`，例如 `300 ns`、`3.725 ms`、`30.081 s`；禁止 `3.73e+03us` 这类科学计数法。
- 整数位加千分位分隔，例如 `5,490`、`230,733`；吞吐使用十进制 `GB/s` 并保留可读精度，例如 `38.76 GB/s`。
- 单样本（n=1）只输出单个值，例如 `3.725 ms`，不输出 `(MAD 0, x..x, n=1)` 噪声。
- 多样本（n>1）输出 `median (MAD m, min..max, n=N)`，其中每个数值同样遵守以上格式。
- `result.json` 中 `logical_throughput_bytes_per_second` 以 byte/s 记录；报告按 `1 GB = 1000³ bytes` 展示，便于和硬件规格直接对比。

## 0. 执行结论

> **留白。** 用 3–5 句话总结本次测试：运行的是完整 canonical suite 还是专项复测；completed/partial/failed/skipped 数量；每个 case 的有效 repetition 数；最重要的 2–3 个发现；是否生成完整 perf 产物；结果能否发布。
>
> 可直接套用以下句式并替换占位符：
>
> 本次在 `<GPU/机器>` 上运行 `<完整 profile suite / 专项 case>`，共 `<N>` 个 case：`<completed>` completed、`<partial>` partial、`<failed>` failed、`<skipped>` skipped，每 case `<N>` 个有效 repetition。最关键的事实是：① `<Tree 观察>`；② `<Transfer/策略观察>`；③ `<磁盘或采样观察>`。代表 case 的 perf 产物 `<完整/不完整>`。`<满足/不满足>`正式发布条件，`<可发布结论/仅用于问题定位>`。

默认只描述事实。只有同时满足以下条件，才可以写“接近硬件上限”“软件无瓶颈”等归因性判断；否则结论统一写“待分析”：

- 全部 canonical case completed，且所引用 repetition 均为 valid。
- Transfer 的 operation、working-set、copy-strategy 和 failure invariants 全部成立。
- 已采集同机 PCIe、内存或磁盘基线，并为对比配置了显式阈值。
- 数据与 perf、vmstat 等旁证方向一致，没有用单一信号替代因果分析。

## 1. 测试环境

GPU、PCIe link、磁盘 device/filesystem、binary/profile SHA 和代码 commit 必须直接使用 suite manifest 的实际采集值，不得手填默认硬件型号或沿用上次报告。

| 项 | 值 |
| --- | --- |
| GPU | `<manifest.environment.gpu>` |
| CPU | `<型号、核数/线程数>` |
| 内存 | `<容量与关键拓扑>` |
| 驱动 / CUDA | `<manifest.environment>` |
| PCIe | `<当前/最大 link generation 与 width；是否 passthrough>` |
| 磁盘 | `<manifest.environment.disk；device、filesystem、挂载方式>` |
| 磁盘基线 | `<同目录 O_DIRECT 顺序读/写 GB/s；命令与文件大小>` |
| Profile | `<profile id + SHA256>` |
| 代码版本 | `<manifest.environment.code_commit>` |
| Binary | `<binary path + SHA256 + build config>` |
| 执行参数 | `<suite/case、cuda-device、perf mode/frequency、repetitions>` |
| 日期 | `<YYYY-MM-DD HH:mm:ss + timezone>` |

> **硬件信息采集命令（每次测试必做）**：
>
> ```bash
> nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv
> lscpu | head -15
> free -g | head -2
> lsblk -d -o NAME,ROTA,SIZE,MODEL,TRAN
> findmnt -T <disk-root>
>
> dd if=/dev/zero of=<disk-root>/w.tmp bs=1M count=4096 oflag=direct
> dd if=<disk-root>/w.tmp of=/dev/null bs=1M count=4096 iflag=direct
> rm -f <disk-root>/w.tmp
> ```
>
> PCIe 理论带宽必须按实际 link generation/width 计算，不要默认写 Gen4 x16。磁盘基线必须使用 benchmark 所在的同一 device/filesystem；共享云盘或虚拟化环境要明确注明。

## 2. Tree 稳态场景

**构造**：两个 case 都使用 scaled payload，先构造 128–768 keys 的共享前缀路径，使树达到 100k nodes。稳态事务按 0.7/0.2/0.1 选择 continuation / fork / cold：continuation 完整 match 已记录路径；fork 复用已有路径 25%–90% 的前缀后分叉；cold 从全新 root path 开始。每次 match 后连续执行 4 次增量 insert，每步把完整 path 再追加 32 keys，最终不超过 1000 keys。第一次 insert 还会一并写入 match 未命中的 fork/cold suffix，因此它实际新增的 nodes 可以大于 32；后续三次通常各新增 32。普通候选在一个 epoch 内无放回，hot subset 另有 20% 抽样概率；候选池有界。路径选择与随机 trace 在计时区外完成。build 后 warmup 10s，measured 至少 30s；`tree_stress_100k` 使用 8 workers，`tree_stress_100k_single` 使用 1 worker。淘汰由 insert 提交触发水位检查，是事件驱动路径，不是后台线程。

| Case | 状态 | duration | insert p50/p99/max | match p50/p99/max | avg matched blocks/request (device/host) | request shape (insert path/new nodes/match keys) | matched depth (continuation/fork/cold) | insert ops/s | match ops/s | loads (committed/succeeded) | 节点水位 avg [min,max] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tree_stress_100k` | | | | | | | | | | | |
| `tree_stress_100k_single` | | | | | | | | | | | |

列语义：

- `duration`：`phases_ns.measured`，只统计 measured window。
- `avg matched blocks/request`：`match_device_matched_blocks_per_request / match_host_matched_blocks_per_request`。
- `request shape`：`insert_path_keys_per_call / insert_new_nodes_per_call / match_keys_per_call`。insert 接口收到的是完整 path；只有 `new nodes` 对应本次实际分配并新增的后缀。
- `matched depth`：continuation / fork / cold 三种事务的实际平均匹配深度；不是配置的计划复用长度。
- insert ops/s：`metrics.insert_calls / phases_ns.measured * 1e9`；match 同理，用于比较多线程吞吐。
- `loads`：`loads_committed / loads_succeeded`；仅当 failed/cancelled/commit failed/pending 非零时追加说明。
- 节点水位：`steady_state_node_count_avg [min, max]`。

load 数与 match 的 host 命中数一致：每次 host 命中触发一次异步 load。两个 case 的 load 数差异主要反映稳态 eviction 将多少 block 推到 host，不能直接判为故障。

### 核心观察

> **留白。** 基于表格和本次火焰图给出 3–4 条分析，至少回答：
>
> 1. 8 workers 相对 1 worker 是否带来 insert/match 吞吐收益，变化幅度是多少？
> 2. 并发对 p50/p99/max 的放大是否与锁竞争、分配、树更新或 eviction 热点一致？
> 3. continuation/fork/cold 的实际 matched depth 是否符合预期？完整 insert path、第一次补齐 miss suffix、后续每步追加 32 keys 的区别是否在结果中成立？
> 4. load 完成情况与节点水位是否稳定，是否存在 failed/cancelled/commit failed/pending 或 trace exhaustion？
>
> 可用句式：`8 workers` 的 insert/match ops/s 相对单线程分别变化 `<x%>/<y%>`；p50/p99 分别变化 `<x×>/<y×>`。perf 的前两项热点为 `<symbol + 占比>`，因此当前证据支持 `<事实性判断>`，但 `<缺少的证据>` 仍需补充。

必须注明 Tree 是 flattened metadata microbenchmark，不复刻 profile 的完整 member fan-out、device-only group 或真实 SWA topology。多线程差异也不能归因于 benchmark-side 采样锁，因为 measured 路径不使用全局采样锁串行化业务调用。

## 3. Transfer 场景

“混合总吞吐”是同一 measured window 内各方向成功字节之和除以墙钟时间，不是两个单方向峰值简单相加。每方向吞吐必须和混合总吞吐一起展示；`mode` 仅 disk case 有值，Device↔Host 填 `-`。

### 3.1 Device↔Host 介质对（d2h+h2d）

四个 case 均为 8 workers，同一 case 在 measured window 内混合执行 d2h+h2d。full_context 与 swa 各自对比 `batch` 和 `staged-sm`；显式 requested strategy 必须严格命中 actual strategy，任何 fallback 都使该 case 无效。

| Case | 状态 | duration | mode | 混合总吞吐（含 d2h/h2d） | ops/s | 总传输 | requested→actual strategy | working set requested/addressable/visited | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `transfer_device_host_full_context_batch` | | | - | | | | `batch →` | | |
| `transfer_device_host_full_context_staged_sm` | | | - | | | | `staged-sm →` | | |
| `transfer_device_host_swa_batch` | | | - | | | | `batch →` | | |
| `transfer_device_host_swa_staged_sm` | | | - | | | | `staged-sm →` | | |

### 3.2 Device↔Disk 介质对（d2disk+disk2d）

device 侧只分配 worker slots，disk 侧分配并寻址完整 working set；相邻 write/read 共享 coordinate。buffered profile working set 固定为 full_context 32768 blocks、swa 4096 blocks，以覆盖完整 addressable working set；direct case 使用 registry 的 auto working set。

| Case | 状态 | duration | mode | 混合总吞吐（含 d2disk/disk2d） | ops/s | 总传输 | requested→actual strategy | working set requested/addressable/visited | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `transfer_device_disk_full_context_direct` | | | direct | | | | | | |
| `transfer_device_disk_full_context_buffered` | | | buffered | | | | | | |
| `transfer_device_disk_swa_direct` | | | direct | | | | | | |
| `transfer_device_disk_swa_buffered` | | | buffered | | | | | | |

### 3.3 Host↔Disk 介质对（h2disk+disk2h）

host 侧只分配 worker slots，disk 侧是完整 addressable working set。buffered working set 与 Device↔Disk 相同，每个 repetition 独立 drain、采样并清理。

| Case | 状态 | duration | mode | 混合总吞吐（含 h2disk/disk2h） | ops/s | 总传输 | requested→actual strategy | working set requested/addressable/visited | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `transfer_host_disk_full_context_direct` | | | direct | | | | | | |
| `transfer_host_disk_full_context_buffered` | | | buffered | | | | | | |
| `transfer_host_disk_swa_direct` | | | direct | | | | | | |
| `transfer_host_disk_swa_buffered` | | | buffered | | | | | | |

### 3.4 硬件带宽极限 vs 实测

| 路径 | 硬件理论/同机实测极限 | benchmark 实测 | 比例 | 结论 |
| --- | --- | --- | ---: | --- |
| Device↔Host | `<按实际 PCIe link 填写>` | `<batch/staged-sm，full_context/swa 分别填写>` | | `<待分析>` |
| Host↔Disk（direct） | `<同 device/filesystem 的 O_DIRECT 读/写基线>` | | | `<待分析>` |
| Device↔Disk（direct） | `<同上，并说明额外 CUDA copy 路径>` | | | `<待分析>` |
| Host↔Disk（buffered） | `<内存/page cache/写回条件>` | | | `<待分析>` |
| Device↔Disk（buffered） | `<page cache + CUDA copy 组合路径>` | | | `<待分析>` |

### 带宽与瓶颈分析

> **留白。** 至少回答以下问题：
>
> 1. full_context/swa 下 `batch` 与 `staged-sm` 的吞吐、ops/s 和方向平衡分别如何；策略差异是否超过多 repetition 波动？
> 2. direct 磁盘路径与同机 O_DIRECT 基线的比例是多少；有无足够证据区分硬件限速、CPU 路径、CUDA copy、同步等待或并发上限？
> 3. buffered 场景是否完整访问 requested/addressable working set，并在 drain/vmstat 中出现相符的写回旁证？
> 4. full_context 与 swa 的单 op payload/tiles 差异如何影响 ops/s、CPU 占用、staging lease 持有时间和混合吞吐？

当 full_context/swa、batch/staged-sm 或 direct/buffered 差异显著时，建议按以下顺序补实验：

1. **CPU 占用对比**：在 measured window 内采样进程和线程 CPU 使用率。接近 0% 通常表示阻塞在 GPU/IO/同步路径，高 CPU 使用率则继续定位用户态热点。
2. **方向平衡**：对比每方向 succeeded bytes/throughput；混合总吞吐正常但单方向失衡时，不要只报告 aggregate。
3. **策略对照**：在同一 GroupSet、并发、working set 和机器上比较 batch/staged-sm，并确认 requested strategy 与 actual strategy 一致。
4. **工作集对照**：用专项复测改变 `--working-set-blocks`，判断 page cache 命中、真实 writeback 和寻址范围对结果的影响；专项结果不能替代 canonical suite。
5. **staging 持约模型**：Device↔Disk 吞吐可用“有效 staging lease 数 / 单 op 持约时间”做交叉检查；当前 runner 会保证 staging 数不少于 worker slot，但仍需用数据验证是否受在飞 op 上限约束。
6. **介质基线与对照路径**：使用同目录 `dd O_DIRECT` 基线，并用 Host↔Disk 对照 Device↔Disk，区分磁盘路径与 CUDA 路径开销。
7. **时间线补证**：CPU perf 不能解释 off-CPU 等待或 GPU copy overlap；必要时补 nsys/NVTX timeline。

### Transfer 发布前不变量

每个 valid repetition 都必须满足：

```text
requested = attempted = succeeded + failed
failed = 0
requested working set = addressable working set = visited working set
explicit canonical requested strategy != auto => requested strategy = actual strategy
requested strategy = auto => actual strategy is recorded for analysis
```

同时核对每方向 attempted/succeeded/failed 与总计数闭环。任何一项不满足，都应标为 partial/failed，并在执行结论中说明，不能把该 repetition 纳入聚合。

## 4. 火焰图与采样质量

`--perf record` 必须为当前 6 个代表 case 生成 `perf.data`、`perf.folded`、`flamegraph.svg` 和 `perf_summary.txt`，并在 suite manifest 的 `perf.artifacts` 中记录相对路径。HTML 必须提供火焰图、原始数据和文本摘要的可点击链接。

| Case | 状态 | 模式 | CPU 火焰图 | 原始数据 | 文本摘要 |
| --- | --- | --- | --- | --- | --- |
| `tree_stress_100k` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |
| `tree_stress_100k_single` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |
| `transfer_device_host_full_context_batch` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |
| `transfer_device_host_full_context_staged_sm` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |
| `transfer_device_disk_full_context_direct` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |
| `transfer_host_disk_full_context_direct` | | record | `<flamegraph.svg link>` | `<perf.data link>` | `<perf_summary.txt link>` |

### 采样质量

| Case | 样本数 | lost samples | `[unknown]` 占比 | event/frequency | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| `<representative perf case>` | | | | | |

> **留白。** 从本次 `perf_summary.txt` 逐 case 填写，不得沿用旧报告。说明 event 是 hardware cycles 还是 cpu-clock、频率是否一致、是否丢样、符号质量是否足以支持热点分析。
>
> `[unknown]` 可能来自闭源库或缺少 frame pointer 的系统组件，但不能在未检查 build config、build-id cache 和 perf.data 属主的情况下直接归因。火焰图只反映 on-CPU 时间；磁盘等待、CUDA 异步传输和流同步等 off-CPU 段不会按等待时长出现在图中。

### 热点汇总

| Case | 前 2 热点（symbol + 占比） | 代码路径/机制 | 结论或待验证假设 |
| --- | --- | --- | --- |
| `tree_stress_100k` | | | |
| `tree_stress_100k_single` | | | |
| `transfer_device_host_full_context_batch` | | | |
| `transfer_device_host_full_context_staged_sm` | | | |
| `transfer_device_disk_full_context_direct` | | | |
| `transfer_host_disk_full_context_direct` | | | |

> **留白。** 热点必须从本次 `perf_summary.txt` 和火焰图读取后填写。优先比较 Tree 多/单线程与 Device↔Host batch/staged-sm 两组对照；磁盘 case 要结合吞吐和 vmstat，不能仅凭 on-CPU 占比判断 IO 瓶颈。

## 5. 系统级 IO 旁证

vmstat 是系统累计量在单 repetition 窗口内的差值，不是单进程、单方向或单次 IO 的精确统计。报告必须明确这一边界。

| Case | pgpgin delta | pgpgout delta | nr_dirty delta | drain seconds |
| --- | ---: | ---: | ---: | ---: |
| `transfer_device_disk_full_context_direct` | | | | |
| `transfer_device_disk_full_context_buffered` | | | | |
| `transfer_device_disk_swa_direct` | | | | |
| `transfer_device_disk_swa_buffered` | | | | |
| `transfer_host_disk_full_context_direct` | | | | |
| `transfer_host_disk_full_context_buffered` | | | | |
| `transfer_host_disk_swa_direct` | | | | |
| `transfer_host_disk_swa_buffered` | | | | |

### vmstat 解读

> **留白。** 必须回答三个问题：
>
> 1. buffered 场景的 dirty/writeback 变化和 drain 时间是否表明内核发生写回或节流；如果缺少阈值采样，应明确证据不足。
> 2. direct 场景的 pgpgin/pgpgout 与逻辑传输量是否数量级一致；不一致时是否可能受单位、系统其他 IO 或采样窗口影响。
> 3. buffered 物理写回量占逻辑写入量的比例如何，是否支持“完整工作集触发真实 writeback”的判断。
>
> 多 repetition 时，表格对各 valid repetition 的窗口差值使用 median/MAD/min/max/n；不要先把不同 repetition 的系统累计值相加再计算单次吞吐。

## 6. 限制与后续建议

> **留白。** 基于本次结果给出 3–5 条，不要列与数据无关的通用口号。可参考以下方向：
>
> - **稳定性**：默认单次 repetition 只能作为单次实测事实；正式性能基线建议每 case ≥5 reps，并比较 median/MAD/min/max。
> - **范围**：专项复测、跳过 disk case 或缺少某种 copy strategy 时，明确哪些结论不能外推到完整 canonical suite。
> - **测量边界**：说明 measured、sync/drain 与 perf 独立 profiling process 的边界，避免把 setup/teardown 或另一次进程的 wall time 混入指标。
> - **分析工具**：perf 只覆盖 on-CPU 视角；磁盘等待、CUDA copy overlap、stream synchronization 需要 nsys/NVTX 或系统 IO 工具补证。
> - **工作集**：说明 canonical buffered working set 是否适合本机内存、page cache 和磁盘容量；如需改变，只能作为有明确标注的专项实验。
> - **优化方向**：结合本次热点和对照实验给出，例如缩短 Tree 锁内 eviction 路径、减少分配/树更新开销、改善 copy tile 调度、方向平衡或 staging 利用率。

## 7. 产物与可复核性

| 产物 | 位置 | 完整性/校验 |
| --- | --- | --- |
| HTML 报告 | `<output-dir>/index.html` | `<可打开；所有相对链接有效>` |
| Suite manifest | `<output-dir>/profile/suite_manifest.json` | `<SHA256>` |
| 原始 repetition 结果 | `<output-dir>/profile/<case>/rep_*/result.json` | `<只列 valid repetitions；数量与 manifest 一致>` |
| stdout/stderr 与采样 | `<output-dir>/profile/<case>/rep_*/` | `<stdout.txt、stderr.txt、vmstat/nvidia-smi before/after>` |
| perf 产物 | `<output-dir>/profile/<case>/perf/` | `<6 个代表 case；data/folded/SVG/summary 齐全>` |
| 原始数据包 | `<OSS URL>/btc_results_<date_time>.tar.gz` | `<SHA256>` |
| 代码 / binary / profile | `<commit + binary SHA256 + profile SHA256>` | `<与 manifest 一致>` |

> **留白。** 在报告末尾记录上传地址、数据包 SHA256、代码 commit、binary SHA256、profile SHA256 和复现命令。所有 HTML 链接使用相对路径，确保 `index.html` 与 `profile/` 一起移动或上传后仍可访问。

---

## 附：如何生成并交付报告

### 1. 构建 perf 版本 binary

```bash
cd github-opensource
bazelisk build -c opt --config=cuda13 --config=sm8x --config=block_tree_benchmark_perf \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_gpu_benchmark \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:block_tree_cache_benchmark_driver
```

必须使用 `--config=block_tree_benchmark_perf`，否则 binary 可能被 strip 或缺少 frame pointer，火焰图符号质量不足。修改 `benchmark_cases.py` 后要重建 driver；修改 C++ runner 后要重建 native binary。

### 2. 准备运行环境

```bash
export LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH
```

确认容器允许 `perf_event_open`；需要时以 `--security-opt seccomp=unconfined` 启动开发容器。不要用 sudo 处理 perf.data，以免 HOME/build-id cache 与文件属主不一致。

### 3. 采集硬件与介质基线

执行 §1 的采集命令，将输出和测试时间一并保存。磁盘空间必须容纳 canonical buffered working set，`--disk-root` 必须指向要测量的真实 device/filesystem。

### 4. 运行完整 profile suite

```bash
block_tree_cache_benchmark_driver \
  --suite profile \
  --perf record \
  --output-dir <output-dir> \
  --disk-root <disk-root> \
  --process-repetitions <N>
```

正式 profile 默认每 case 1 个 repetition，约 25–35 分钟；只有明确需要稳定性分析时才提高 `<N>`。单 case 或 case 列表是专项复测，报告必须明确范围。

### 5. 生成 HTML 报告

```bash
python3 rtp_llm/cpp/cache/block_tree_cache/benchmark/generate_report.py \
  --output-dir <output-dir> \
  --output <output-dir>/index.html
```

脚本只读取 `<output-dir>/profile/suite_manifest.json` 中的 valid repetitions，并自动生成环境、状态、Tree/Transfer、带宽、perf 和 vmstat 表格。

### 6. 补齐留白并复核

按本模板补齐 §0、§2、§3、§4、§5、§6、§7 的分析与清单，然后逐项检查：

- suite 完整性与发布范围表述一致。
- 14 个当前 profile case 均按 manifest 展示；专项报告缺失的类别明确标注“本次无此类 case”。
- 6 个代表 perf case 的链接可打开，采样质量已说明。
- 所有数值来自本次 valid repetitions，格式无科学计数法。
- working-set、strategy、failure invariants 已逐 repetition 验证。
- 所有“硬件限速”“接近上限”等结论都有同机基线和旁证。

### 7. 离线重建火焰图（需要时）

driver 在 `record` 模式下会自动生成火焰图。需要从已有 perf.data 重新生成时：

```bash
perf script -i <perf.data> > perf.unfold
<FlameGraph-dir>/stackcollapse-perf.pl perf.unfold > perf.folded
<FlameGraph-dir>/flamegraph.pl perf.folded > flamegraph.svg
```

`perf script` 必须以 perf.data 属主身份执行。重建后保持文件名为 `flamegraph.svg`，并检查 manifest/HTML 中的相对链接。

### 8. 打包与上传

```bash
tar czf btc_results_<date_time>.tar.gz -C <output-dir> index.html profile/
sha256sum btc_results_<date_time>.tar.gz

PREFIX="$(whoami)/$(date +%Y%m%d_%H%M%S)"
ossutil cp -r <output-dir>/ oss://search-ad/$PREFIX/
ossutil cp btc_results_<date_time>.tar.gz oss://search-ad/$PREFIX/
curl -sI "http://search-ad.oss-cn-hangzhou-zmf.aliyuncs.com/$PREFIX/index.html" | head -3
```

覆盖已存在的 OSS 路径时，`ossutil cp` 会交互式询问；非 TTY 下可能静默跳过。确需覆盖时显式加 `-f`，上传后必须用 HTTP 请求核验。

### 常见陷阱

| 陷阱 | 症状 | 解决 |
| --- | --- | --- |
| 未设置 `LD_LIBRARY_PATH` | `libpython3.10.so.1.0: cannot open shared object file` | 加入 `/opt/conda310/lib` 后重试 |
| 未用 perf config 构建 | 火焰图大量 `[unknown]` | 使用 `--config=block_tree_benchmark_perf` 重建 |
| 容器 seccomp 拦截 perf | `perf.data` 为空或小于 1 KB | 使用允许 `perf_event_open` 的容器配置 |
| 以 sudo 执行 perf script | 符号不全、build-id cache 未命中 | 以 perf.data 属主身份执行 |
| native process 未输出 `MEASURE_START` | perf 收集报 attach/marker 失败 | runner 在 measured 前输出 marker 并留出 attach 时间 |
| 忘记 `--disk-root` | disk case 为 `skipped_no_disk` | 指向真实且容量足够的磁盘目录 |
| driver/native binary 未重建 | 参数或行为与源码不一致 | Python registry 改动后重建 driver；C++ 改动后重建 binary |
| Device↔Disk 报 staging pool exhausted | 旧 binary 的 staging 数小于并发 | 使用最新版 runner 并重建 binary |
| 把 mixed throughput 当单向峰值 | 报告带宽超过物理含义或比较错误 | 同时展示 aggregate 与各方向吞吐 |
| 混入历史 result.json | 样本数或数值与 manifest 不一致 | 只读取 manifest 标记的 valid repetitions |
| 照抄旧报告热点/结论 | 结论与本次数据不符 | 从本次 perf_summary、火焰图、vmstat 和基线重新分析 |

更多运行细节见 `../README.md`，case 构造和当前 registry 见 `benchmark_cases.md`。
