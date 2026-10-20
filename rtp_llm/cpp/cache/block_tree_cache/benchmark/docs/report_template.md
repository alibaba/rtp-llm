# BlockTreeCache Benchmark 报告指南

本文是写作参考，供人工或 agent 根据实验目的组织报告。章节顺序、表头、图形、篇幅和输出格式均可调整；可以使用 Markdown、HTML 或其他方便分享的形式。仓库不提供固定报告生成器，新增 case 不需要修改渲染代码。case 家族和选择方式见 [benchmark_cases.md](benchmark_cases.md)。

## 基本信息

一份可理解、可复核的报告应交代以下内容，表达方式不限：

- **实验目的与范围**：想回答什么问题，实际选择了哪些 suite/case/矩阵维度，是否只展示部分结果；说明完成、失败、跳过或缺失情况。可以按家族汇总并链接完整清单，无需逐个展示全部 case，但不要把部分成功描述为整个 suite 或全部 profiling 完整。
- **环境与配置**：与结论有关的 GPU、CPU、磁盘和软件版本，代码 commit、binary/profile 指纹，以及关键实际参数和运行命令。信息来自本次 manifest/result；容器可见的磁盘信息不足以推断宿主物理盘。
- **结果与口径**：关键指标、单位、测量窗口、有效样本数，以及必要的比较条件。聚合使用本次 manifest 标记 valid 且 completed 的 repetition，并确认对应 result completed；失败、跳过、旧文件和独立 profiler 进程不混入有效样本。
- **观察与限制**：哪些是数据直接支持的观察，哪些是推测；比较中还有哪些变量发生变化，哪些证据尚未采集。没有同机硬件基线时，不直接断言接近硬件上限。
- **数据来源**：提供 suite manifest、原始 result 和相关日志/图形的位置或链接，让读者能追溯汇总和未展开的数据。

单次测量可以直接展示数值并标注 n=1；多次测量建议给出 median 和离散程度（如 MAD 或 min/max），不要只挑最好的一次。数值建议使用易读的单位。上述要求约束数据解释，不规定页面布局。

## 按实验选择展示方式

| 实验 | 可选展示方式 | 理解结果所需的背景 |
| --- | --- | --- |
| Tree 在线生命周期 | 请求完成数/req/s、关键时延表，或 task-pool 对照图 | C32 是逻辑 context；单 foreground scheduler；固定模拟 forward sleep；block/token 与 scaled payload；BASE/CONTINUATION 请求构造 |
| Mixed Transfer | 按介质、group set、strategy 展示吞吐和 batch 信息 | 混合总吞吐是同一窗口各方向成功字节之和除以墙钟时间，不是单方向峰值相加 |
| Batch API matrix | 按方向和 group set 比较 C/D 提交粒度及 copy strategy | transfer-engine descriptor batch 与 CUDA batch copy strategy 是两层概念；以实际参数为准 |
| E2E business matrix | 上层业务并发 × 下层 worker 数的表格或热力图 | 标明单方向、group set、每业务 descriptor 数和 batch；不等同于模型推理 e2e |
| 专项子集或新增 case | 围绕问题自选表格、图形和分析 | 说明选择范围、负载语义、变量与指标口径即可 |

Tree 的 req/s 可由 `completed_request_transactions / phases_ns.measured * 1e9` 计算，表示含模拟 forward 的 benchmark 生命周期速率，不是线上模型 TPS。任务池对照应核对除 pool size 外的配置、profile、seed/repetition、trace、binary SHA 和代码 commit；有差异时说明可比性的限制。水位观察和 dependency skip 等诊断项可以按需解释，不应被当作未发生的运行失败。

Transfer 可以按需展示各方向吞吐、ops/s、传输字节、requested/actual strategy、实际 descriptor batch、working set 与失败数。无需固定三个介质分表；case 的方向与介质依据实际配置和结果，不靠名称子串猜测。

## Profiling 与原始产物

根据分析问题链接有关的 perf、off-CPU、nsys 或其他产物，交代独立采集的配置和采样质量。正式 profile 的采集流程及有效性条件见 [README](../README.md) 的 profiling 说明；专项可按目的选择采集范围，并说明缺项和原因。suite completed 与 profiling 完整是不同概念，未采集不能写成“无热点”。

固定 forward sleep 和 task-pool idle 是 Tree 的预期行为；额外锁等待、load/evict 等待或 scheduler no-ready wait 才是需要结合证据分析的线索。`pgpgin`/`pgpgout` 是系统窗口差值，不能精确归因到单进程或单方向 IO。

常见产物位置供参考，实际以本次 manifest 和采集记录为准：

| 产物 | 常见位置 |
| --- | --- |
| Suite manifest | `profile/suite_manifest.json`（smoke 对应 `smoke/`） |
| Repetition 结果和日志 | `profile/<case>/rep_*/` |
| perf 数据和火焰图 | `profile/<case>/perf/` |
| off-CPU 数据和采集记录 | 独立采集目录，如 `profile/<case>/offcpu_<RUN_ID>/` |
| 分析报告 | 自选文件名和格式，与原始数据一起保存或链接 |

可以先给结论，再解释实验和证据，也可以按多个实验问题分别展开。共享报告时保留有效的数据链接即可，不要求生成 `index.html` 或采用本文的章节顺序。
