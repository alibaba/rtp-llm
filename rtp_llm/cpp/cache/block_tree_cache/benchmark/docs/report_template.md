# BlockTreeCache Benchmark Report Template

## 1. 测试环境

GPU、CPU、内存、kernel、binary/profile SHA 与代码 commit 均来自 suite manifest 的实际采集。原始 manifest 保留 `disk_target`、mount source/fstype、容量和 mount namespace 口径；HTML 环境表将这些信息压成一行 `disk`，不重复展示 target，也不再追加整段磁盘说明。

Docker 中该行只标记为 `container-visible mount namespace`，不输出完整 overlay `lowerdir/upperdir`，也不猜测宿主物理块设备。若目标是测宿主某块实际磁盘，先 bind mount 对应目录，再把 `--disk-root` 指向容器内路径；详细信息由 suite manifest 和复现命令承载。

## 2. Tree 在线生命周期场景

### 测试构造

- C32 表示 32 个逻辑会话 context（数据对象，不是线程），由 1 个前台 scheduler 推进；task pool size 只控制后台 cache load/evict/store 线程。
- Tree 固定每个逻辑 path block 为 256 tokens。fixture 使用 scaled payload：报告必须列出每个 GroupSet 的 `scaled_payload_bytes`，并说明这不是线上模型每 block 的实际显存占用。
- 初始 topology 是 3,711-block shared base + 16,289-block background tree；每个 GroupSet 的 device/host pool 各 32,768 blocks。
- trace 有两种请求：BASE（新 epoch，从 shared base 选择计划命中前缀后追加唯一 suffix）和 CONTINUATION（继承同 family 的父 path，只追加更长的唯一 tail）。每个 family 首次请求或抽样长度不大于当前 leaf 时开始新的 BASE；只有抽样长度更大时才生成 CONTINUATION。
- 请求长度从 20 个 token 桶按权重抽样；报告用几句话概括长度与逻辑 block 范围、主要长度桶及其合计占比，不再展开 20 行明细表。BASE 的计划前缀命中率从 `0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99%` 这 13 个等概率档位抽样。
- 完整请求生命周期为 `match → async load（如需要）→ READY batch forward → insert full path → release refs`。每个 READY batch 固定 100ms 模拟 forward；warmup 15s，measured 60s。

### Tree 表

| Case | 状态 | 后台 cache 任务线程 | 测量窗口 | 已完成请求生命周期（数量 / 每秒） | 请求组成（新会话 / 续写） | 命中深度（计划 / 实际） | cache 查找时延 | 路径发布时延 | 查找到可 forward 时延 | READY batch | 结束清理 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

表内“每秒”是完整请求生命周期数除以 measured 墙钟时间，单位写作 `req/s`，不是线上模型 TPS。水位快照、依赖跳过等诊断项移到“主要观察”，不使用红色失败徽标展示非失败观察值。

### Tree 主要观察

- `tree_online_high_variation_c32`：C32 逻辑 context，前台 scheduler 1 线程，task pool 4/8。trace hash `<code>`。forward N batch / M requests。模拟 forward sleep = batches × 100ms。
- 生命周期：active peak ≤ C32，loading peak，load tickets pending peak，request-held blocks peak（跨 forward 持有），unexpected extra match；后三类压力/匹配量是观察项。
- 请求组成：用“新会话/续写”解释 BASE/CONTINUATION；用“因前序请求尚未发布而暂缓 admission 的扫描次数”解释 dependency skip，不把内部字段名当作表头。
- warmup 后水位是负载形态观察值，不是 PASS 条件；若未满足，应写明具体阈值及边界，不能显示为红色 `fail/not ready`。
- finalize 的 active requests、pending tickets、pending tasks、REQUEST refs 全部为 0，drain timeout 为 0。

### On-CPU 热点

- BlockTreeCache match/insert 锁竞争
- eviction 淘汰路径 CPU 占比
- load 异步流水线效率

### Off-CPU

- 正式 profile 默认必须采集 `tree_online_high_variation_c32` 的 15 秒 BCC off-CPU；它不是用户额外点名后才添加的可选项。
- 只有 README 规定的 sidecar、namespace、内核能力、权限、BCC smoke 或符号质量预检失败时才允许 skip，并必须展示具体失败项。`not requested`、`用户未单独要求` 或 `只要求 benchmark` 不是合法 skip 原因。
- 环境预检通过但没有有效 folded、SVG 和 manifest 时，报告必须标记为 profiling 不完整，不得以“canonical suite completed”掩盖缺失产物。
- 固定 100ms forward sleep（sleep_for/nanosleep）是预期 off-CPU 时间，不归因成 BlockTreeCache 退化
- task-pool idle 时 condition-variable wait 是预期背景
- load/evict transfer 的 futex/IO 等待

## 3. Transfer 场景

"混合总吞吐"是同一 measured window 内各方向成功字节数之和除以墙钟时间，不是两个单方向峰值相加。

`descriptor batch` 表示一次 transfer-engine `submit(vector<TransferDescriptor>)` 携带的 descriptor 数；`strategy=batch` 表示 Device↔Host 执行器使用 CUDA batch copy。二者必须分列，不能互相替代。Device→Disk 目前只支持 singleton，因此该方向实际 avg/max 应为 `1 / 1`；另外五个方向应展示真实批量值。

### Device↔Host 介质对

| Case | 状态 | duration | mode | 混合总吞吐（含各方向） | ops/s | 总传输 | requested → actual strategy | descriptor batch API requested → resolved；各方向 avg / max | working set | failed |

### Device↔Disk 介质对

| Case | 状态 | duration | mode | 混合总吞吐（含各方向） | ops/s | 总传输 | requested → actual strategy | descriptor batch API requested → resolved；各方向 avg / max | working set | failed |

### Host↔Disk 介质对

| Case | 状态 | duration | mode | 混合总吞吐（含各方向） | ops/s | 总传输 | requested → actual strategy | descriptor batch API requested → resolved；各方向 avg / max | working set | failed |

### 带宽判断

| 路径 | 硬件理论/实测极限 | benchmark 实测 | 结论 |
| --- | --- | --- | --- |
| Device↔Host | PCIe Gen4 x16 理论单向 ~32 GB/s、双向 ~64 GB/s | - | 待分析 |
| Host↔Disk (direct) | 云盘 O_DIRECT 实测带宽 | - | 待分析 |
| Device↔Disk (direct) | 云盘 O_DIRECT 实测带宽 | - | 待分析 |
| Host↔Disk (buffered) | page cache 吸收后受 host 内存带宽约束 | - | 待分析 |
| Device↔Disk (buffered) | page cache + CUDA copy 组合路径 | - | 待分析 |

## 4. 火焰图与采样质量

表格只列出实际生成 profiling artifact 的 case。`skipped`、`not collected` 或没有 SVG/perf.data/folded 的 case 不占表格行，但 Off-CPU 小节始终存在：无产物时必须展示 manifest 或 `report_metadata.offcpu_status` 提供的**环境预检失败原因**，不允许静默隐藏。正式 profile 若既无有效 off-CPU 产物、又无合规的环境 preflight skip 证据，报告状态必须写成“profiling 不完整，不可发布”。

| Case | 状态 | 模式 | CPU 火焰图 | 原始数据 | 文本摘要 |
| --- | --- | --- | --- | --- | --- |

| Case | 状态 | raw folded | Off-CPU SVG | manifest | 质量摘要 |
| --- | --- | --- | --- | --- | --- |

## 5. 产物与可复核性

| Case | 位置 | 完整性/校验 |
| --- | --- | --- |
| HTML 报告 | index.html | 检查所有相对链接 |
| Suite manifest | profile/suite_manifest.json | 记录 case、repetition 与环境指纹 |
| 原始 repetition 结果 | profile/<case>/rep_*/result.json | 只使用 valid repetitions |
| stdout/stderr | profile/<case>/rep_*/ | stdout、stderr、vmstat/nvidia-smi |
| perf 产物 | profile/<case>/perf/ | perf.data、folded、SVG、summary |
| off-CPU 产物 | profile/<case>/offcpu/ | folded、SVG、manifest |
